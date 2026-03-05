# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
Optimized Pathology Inference with GPU Acceleration
Uses PyTorch DataLoader for efficient parallel data loading and batched GPU inference.
"""

import os
import time
import cv2
import numpy as np
import torch
import csv
from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import TorchVisionFCModel

# Try to import cucim/cupy, fall back to openslide
try:
    import cupy as cp
    from hipcim import CuImage
    USE_CUCIM = True
    print("Using cuCIM/HIPcuCIM for image loading (GPU-accelerated)")
except ImportError:
    try:
        import openslide
        USE_CUCIM = False
        print("cuCIM not available, using OpenSlide")
    except ImportError:
        raise ImportError("Neither cuCIM/HIPcuCIM nor OpenSlide is available")


class WSITileDataset(Dataset):
    """
    PyTorch Dataset for extracting tiles from a Whole Slide Image.

    Each worker in the DataLoader extracts and preprocesses tiles independently,
    enabling parallel CPU-side data loading while the GPU runs inference on the
    previous batch. This is the primary source of speedup over sequential loading.
    """

    def __init__(self, wsi_path, tile_size=224, stride=224):
        self.tile_size = tile_size
        self.stride = stride
        self.wsi_path = str(wsi_path)

        # Load WSI to read dimensions
        if USE_CUCIM:
            self.img = CuImage(self.wsi_path)
            self.height, self.width = self.img.shape[:2]
        else:
            self.slide = openslide.OpenSlide(self.wsi_path)
            self.width, self.height = self.slide.dimensions

        # Pre-calculate all valid tile positions (skip partial edge tiles)
        self.tile_coords = []
        for y in range(0, self.height - tile_size + 1, stride):
            for x in range(0, self.width - tile_size + 1, stride):
                self.tile_coords.append((x, y))

        print(f"WSI Dimensions: {self.width} x {self.height}")
        print(f"Total tiles: {len(self.tile_coords)}")

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        x, y = self.tile_coords[idx]

        # Extract tile - optimized path avoids unnecessary PIL conversions
        if USE_CUCIM:
            tile = self.img.read_region(
                location=(x, y),
                size=(self.tile_size, self.tile_size),
                level=0
            )
            tile_np = cp.asnumpy(tile)

            # Only resize if tile_size != 224 (model input size)
            if self.tile_size != 224:
                tile_pil = Image.fromarray(tile_np).convert("RGB")
                tile_array = np.array(tile_pil.resize((224, 224), Image.Resampling.LANCZOS))
            else:
                tile_array = tile_np if tile_np.shape[-1] == 3 else tile_np[:, :, :3]
        else:
            tile_pil = self.slide.read_region(
                (x, y), 0,
                (self.tile_size, self.tile_size)
            ).convert("RGB")

            if self.tile_size != 224:
                tile_array = np.array(tile_pil.resize((224, 224), Image.Resampling.LANCZOS))
            else:
                tile_array = np.array(tile_pil)

        # HWC -> CHW format for PyTorch
        if tile_array.ndim == 3 and tile_array.shape[2] == 3:
            tile_chw = np.transpose(tile_array, (2, 0, 1))
        else:
            tile_chw = tile_array

        # Normalize: [0, 255] -> [-1, 1] in a single fast operation
        # Equivalent to ScaleIntensityRanged(a_min=0, a_max=255, b_min=-1, b_max=1)
        tile_normalized = (tile_chw.astype(np.float32) / 127.5) - 1.0

        return {
            'image': torch.from_numpy(tile_normalized),
            'image_raw': tile_chw,  # Keep raw CHW uint8 for UI patch display
            'coords': (x, y),
            'idx': idx
        }

    def close(self):
        if not USE_CUCIM and hasattr(self, 'slide'):
            self.slide.close()


def collate_fn(batch):
    """Custom collate function for DataLoader - stacks images into a batch tensor."""
    images = torch.stack([item['image'] for item in batch])
    raw_images = [item['image_raw'] for item in batch]
    coords = [item['coords'] for item in batch]
    indices = [item['idx'] for item in batch]
    return {
        'images': images,
        'raw_images': raw_images,
        'coords': coords,
        'indices': indices
    }


def create_memory_efficient_thumbnail(img, width, height, scale_factor):
    """Create a thumbnail by sampling tiles across the image to avoid loading full resolution"""
    print("Creating memory-efficient thumbnail...")

    target_width = int(width * scale_factor)
    target_height = int(height * scale_factor)

    tiles_x = min(20, target_width // 10)
    tiles_y = min(20, target_height // 10)

    tile_width = width // tiles_x
    tile_height = height // tiles_y

    thumbnail = Image.new('RGB', (target_width, target_height), (255, 255, 255))

    target_tile_width = target_width // tiles_x
    target_tile_height = target_height // tiles_y

    for y in range(tiles_y):
        for x in range(tiles_x):
            try:
                src_x = x * tile_width
                src_y = y * tile_height

                tile = img.read_region(location=(src_x, src_y),
                                       size=(tile_width, tile_height), level=0)
                tile_np = cp.asnumpy(tile)
                tile_pil = Image.fromarray(tile_np).convert("RGB")

                tile_resized = tile_pil.resize((target_tile_width, target_tile_height), Image.Resampling.LANCZOS)

                dst_x = x * target_tile_width
                dst_y = y * target_tile_height

                thumbnail.paste(tile_resized, (dst_x, dst_y))

            except Exception as e:
                print(f"Warning: Failed to sample tile at ({src_x}, {src_y}): {e}")
                continue

    print(f"Memory-efficient thumbnail created: {tiles_x}x{tiles_y} tiles sampled")
    return thumbnail


def main(
    wsi_path="demo/data/pathology_tumor_detection/tumor_001.tif",
    output_dir="demo/output",
    model_path=None,
    model_dir="demo/custom_trained/pathology_tumor_detection/models",
    tile_size=224,
    stride=None,
    batch_size=512,
    num_workers=16,
    scale_factor=0.04,
    device_id=0,
    multi_gpu=False,
    gpu_ids=None,
    confidence_threshold=0.5,
    progress_callback=None
):
    """
    Optimized pathology inference using PyTorch DataLoader for parallel data loading.

    Key optimizations over the sequential approach:
    - DataLoader with multiple workers for parallel tile extraction on CPU
    - Pin memory + non-blocking GPU transfer for overlapped data movement
    - Prefetching to keep the GPU fed with data
    - cuDNN benchmark + TF32 for faster convolutions
    - GPU warm-up to avoid first-batch latency
    - Larger default batch size (512) to maximize GPU utilization
    - Simplified preprocessing (direct numpy ops instead of MONAI Compose pipeline)
    """
    if stride is None:
        stride = tile_size
    Image.MAX_IMAGE_PIXELS = None

    # Import path utilities for consistent path handling
    import sys
    from pathlib import Path as PathLib
    current_file = PathLib(__file__).resolve()
    project_root = None
    for parent in current_file.parents:
        if parent.name == "rocm-ls-examples":
            project_root = parent
            break
    if project_root:
        sys.path.insert(0, str(project_root))

    from components.path_utils import resolve_path

    def resolve_path_wrapper(path_str):
        """Convert path to absolute Path object using utility"""
        return Path(resolve_path(path_str))

    # Convert paths to absolute Path objects
    wsi_path = resolve_path_wrapper(wsi_path)
    output_dir = resolve_path_wrapper(output_dir)
    model_dir = resolve_path_wrapper(model_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    output_image_path = output_dir / "patho_test_output_img.png"
    output_csv_path = output_dir / "patho_test_output.csv"

    # Performance tracking
    start_time = time.time()
    image_name = Path(wsi_path).stem

    # =========================================================================
    # DEVICE CONFIGURATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("DEVICE CONFIGURATION")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if multi_gpu:
        if gpu_ids is not None:
            if isinstance(gpu_ids, str):
                gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        else:
            gpu_ids = list(range(num_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using Multi-GPU: {gpu_ids}")
    else:
        if device_id >= num_gpus:
            print(f"Warning: Requested device {device_id} not available. Using device 0.")
            device_id = 0
        device = torch.device(f'cuda:{device_id}')
        gpu_ids = [device_id]
        print(f"Using Single GPU: {device_id}")

    for gid in gpu_ids:
        print(f"  GPU {gid}: {torch.cuda.get_device_name(gid)}")
        mem_gb = torch.cuda.get_device_properties(gid).total_memory / 1024**3
        print(f"    Memory: {mem_gb:.2f} GB")

    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL LOADING")
    print("=" * 80)

    model_load_start = time.time()

    # Determine which model to use
    if model_path is not None:
        model_path_str = str(model_path)
        if not os.path.exists(model_path_str):
            raise FileNotFoundError(f"Specified model file not found: {model_path_str}")
        print(f"Using selected model: {model_path_str}")
        final_model_path = model_path_str
    else:
        model_files = []
        model_extensions = ['.pt', '.pth', '.pkl', '.ts']
        for ext in model_extensions:
            model_files.extend(Path(model_dir).glob(f"*{ext}"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        final_model_path = str(model_files[0])
        print(f"Using discovered model: {final_model_path}")

    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=False)
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    # Multi-GPU wrapping
    if multi_gpu and len(gpu_ids) > 1:
        print(f"Wrapping with DataParallel on GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # =========================================================================
    # GPU OPTIMIZATIONS
    # =========================================================================
    # cuDNN benchmark: auto-tunes convolution algorithms for the input size
    torch.backends.cudnn.benchmark = True
    # TF32: uses TensorFloat-32 for faster matrix math on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("PyTorch optimizations enabled (cuDNN benchmark, TF32)")

    # GPU warm-up: pre-compiles CUDA kernels to avoid first-batch latency
    print("Warming up GPU...")
    warmup_batch = min(batch_size, 64)  # Use smaller batch for warm-up
    dummy = torch.randn(warmup_batch, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    torch.cuda.synchronize()
    del dummy
    torch.cuda.empty_cache()
    print("GPU warm-up complete")

    # =========================================================================
    # DATA LOADING (DataLoader with parallel workers)
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATA LOADING")
    print("=" * 80)

    dataset = WSITileDataset(wsi_path, tile_size=tile_size, stride=stride)
    expected_total_tiles = len(dataset)
    width, height = dataset.width, dataset.height

    # Scale workers for multi-GPU
    actual_workers = num_workers
    if multi_gpu and len(gpu_ids) > 1:
        actual_workers = num_workers * len(gpu_ids)
        print(f"Multi-GPU: scaling workers from {num_workers} to {actual_workers}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        collate_fn=collate_fn,
        pin_memory=True,             # Faster CPU->GPU transfer via page-locked memory
        prefetch_factor=4 if actual_workers > 0 else None,  # Prefetch 4 batches per worker
        persistent_workers=True if actual_workers > 0 else False  # Keep workers alive
    )

    print(f"DataLoader configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {actual_workers}")
    print(f"  Pin memory: True")
    print(f"  Prefetch factor: {4 if actual_workers > 0 else 'N/A'}")
    print(f"  Persistent workers: {actual_workers > 0}")

    # =========================================================================
    # INFERENCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("INFERENCE")
    print("=" * 80)

    predictions = []
    tumor_coordinates = []
    tiles_processed = 0
    tumor_count = 0
    batch_times = []

    # Patch tracking for UI display
    tumor_patches = []
    normal_patches = []
    all_recent_patches = []

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()

            # Move to GPU with non-blocking transfer (overlaps with compute when pin_memory=True)
            images = batch['images'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Process results on CPU
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
            pred_classes = (probabilities > confidence_threshold).astype(int)
            coords = batch['coords']
            raw_images = batch['raw_images']

            # Store results
            for i, (prob, pred_class) in enumerate(zip(probabilities, pred_classes)):
                x, y = coords[i]
                predictions.append(((x, y), int(pred_class), float(prob)))
                tiles_processed += 1

                # Build patch data for UI
                patch_data = {
                    'image': raw_images[i],
                    'prob': float(prob),
                    'coords': (x, y),
                    'label': 'Tumor' if pred_class == 1 else 'Normal'
                }

                if pred_class == 1:
                    tumor_count += 1
                    tumor_coordinates.append((x, y, tile_size, tile_size))
                    tumor_patches.append(patch_data)
                    all_recent_patches.append(patch_data)

                    # Send immediate update on tumor detection
                    if progress_callback:
                        import random
                        shuffled_patches = all_recent_patches.copy()
                        random.shuffle(shuffled_patches)
                        display_patches = shuffled_patches[:10]

                        elapsed_time = time.time() - inference_start
                        true_tiles_per_sec = tiles_processed / elapsed_time if elapsed_time > 0 else 0
                        eta_seconds = (expected_total_tiles - tiles_processed) / true_tiles_per_sec if true_tiles_per_sec > 0 else 0

                        progress_callback({
                            'type': 'progress',
                            'tiles_processed': tiles_processed,
                            'total_tiles': expected_total_tiles,
                            'tumor_count': tumor_count,
                            'tiles_per_sec': true_tiles_per_sec,
                            'eta_minutes': eta_seconds / 60,
                            'display_patches': display_patches
                        })
                else:
                    normal_patches.append(patch_data)
                    all_recent_patches.append(patch_data)

                # Keep only recent 50 patches for dynamic shuffling
                if len(all_recent_patches) > 50:
                    all_recent_patches.pop(0)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - inference_start
                tiles_per_sec = tiles_processed / elapsed if elapsed > 0 else 0
                eta_sec = (expected_total_tiles - tiles_processed) / tiles_per_sec if tiles_per_sec > 0 else 0

                progress_msg = (
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Tiles: {tiles_processed}/{expected_total_tiles} "
                    f"({tiles_processed/expected_total_tiles*100:.1f}%) | "
                    f"Speed: {tiles_per_sec:.1f} tiles/s | "
                    f"ETA: {eta_sec/60:.1f}min | "
                    f"Tumors: {tumor_count}"
                )
                print(progress_msg)

                # Send periodic progress update to UI
                if progress_callback:
                    import random
                    shuffled_patches = all_recent_patches.copy()
                    random.shuffle(shuffled_patches)
                    display_patches = shuffled_patches[:10]

                    progress_callback({
                        'type': 'progress',
                        'tiles_processed': tiles_processed,
                        'total_tiles': expected_total_tiles,
                        'tumor_count': tumor_count,
                        'tiles_per_sec': tiles_per_sec,
                        'eta_minutes': eta_sec / 60,
                        'display_patches': display_patches
                    })

    inference_time = time.time() - inference_start

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total tiles processed: {tiles_processed}")
    print(f"Tumor tiles: {tumor_count}")
    print(f"Normal tiles: {tiles_processed - tumor_count}")

    tumor_percentage = (tumor_count / tiles_processed) * 100 if tiles_processed > 0 else 0
    print(f"Tumor percentage: {tumor_percentage:.2f}%")

    overall_classification = 'TUMOR DETECTED' if tumor_percentage > 50 else 'PREDOMINANTLY NORMAL'
    print(f"Overall classification: {overall_classification}")

    print(f"\nInference time: {inference_time:.2f}s")
    print(f"Throughput: {tiles_processed/inference_time:.2f} tiles/s")
    print(f"Avg batch time: {np.mean(batch_times):.3f}s")

    # GPU memory report
    print(f"\nGPU Memory Usage:")
    for gid in gpu_ids:
        alloc = torch.cuda.memory_allocated(gid) / 1024**3
        cached = torch.cuda.memory_reserved(gid) / 1024**3
        print(f"  GPU {gid}: {alloc:.2f} GB allocated, {cached:.2f} GB cached")

    # =========================================================================
    # SAVE CSV
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    print(f"Saving prediction results to CSV: {output_csv_path}")
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'x_coordinate', 'y_coordinate', 'prediction_class', 'prediction_probability'])
        for (x, y), pred_class, prob in predictions:
            csv_writer.writerow([image_name, x, y, pred_class, prob])
    print(f"CSV saved with {len(predictions)} predictions: {output_csv_path}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("Creating visualization...")
    viz_start = time.time()

    target_w = int(width * scale_factor)
    target_h = int(height * scale_factor)

    try:
        if USE_CUCIM:
            # Use pyramid level for efficient downscaling
            num_levels = dataset.img.resolutions['level_count']
            best_level = min(num_levels - 1, 4)
            level_dims = dataset.img.resolutions['level_dimensions'][best_level]
            print(f"Using pyramid level {best_level}: {level_dims}")

            downscaled = dataset.img.read_region(location=(0, 0), size=level_dims, level=best_level)
            img_np = cp.asnumpy(downscaled)
            full_image_pil = Image.fromarray(img_np).convert("RGB")
            full_image_pil = full_image_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            full_image_pil = dataset.slide.get_thumbnail((target_w, target_h))
            if full_image_pil.size != (target_w, target_h):
                full_image_pil = full_image_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Pyramid/thumbnail method failed ({e}), using tile sampling fallback")
        if USE_CUCIM:
            full_image_pil = create_memory_efficient_thumbnail(dataset.img, width, height, scale_factor)
        else:
            # Simple fallback for openslide
            full_image_pil = Image.new('RGB', (target_w, target_h), (255, 255, 255))

    actual_width, actual_height = full_image_pil.size
    actual_scale_x = actual_width / width
    actual_scale_y = actual_height / height

    print(f"Output image size: {actual_width}x{actual_height}")

    # Draw tumor annotations
    draw = ImageDraw.Draw(full_image_pil)
    rectangle_width = max(1, min(5, int(3 / scale_factor)))

    annotations_drawn = 0
    for coord_x, coord_y, coord_w, coord_h in tumor_coordinates:
        scaled_x = int(coord_x * actual_scale_x)
        scaled_y = int(coord_y * actual_scale_y)
        scaled_w = max(1, int(coord_w * actual_scale_x))
        scaled_h = max(1, int(coord_h * actual_scale_y))

        if (scaled_x + scaled_w >= actual_width or scaled_y + scaled_h >= actual_height or
                scaled_x < 0 or scaled_y < 0):
            continue

        draw.rectangle([scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h],
                       outline="red", width=rectangle_width)
        annotations_drawn += 1

    print(f"Annotations drawn: {annotations_drawn}/{len(tumor_coordinates)}")
    viz_time = time.time() - viz_start
    print(f"Visualization created in {viz_time:.2f}s")

    # Convert and save image
    img_np = np.array(full_image_pil)
    img_small_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.shape[2] == 3 else img_np

    success = cv2.imwrite(str(output_image_path), img_small_rgb)
    if success:
        print(f"Image saved: {output_image_path}")
    else:
        print("Failed to save image")

    # For Streamlit UI, use RGB format
    plot_img = cv2.cvtColor(img_small_rgb, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_small_rgb

    # Cleanup
    dataset.close()

    # =========================================================================
    # RETURN RESULTS (preserving API contract with driver.py / Streamlit UI)
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s")

    result = {
        'predictions': predictions,
        'tumor_count': tumor_count,
        'total_tiles': tiles_processed,
        'tumor_percentage': tumor_percentage,
        'overall_classification': overall_classification,
        'processing_time': inference_time,
        'total_time': total_time,
        'tiles_per_second': tiles_processed / inference_time if inference_time > 0 else 0,
        'output_image_path': str(output_image_path),
        'output_csv_path': str(output_csv_path),
        'output_image': plot_img,
        'thumbnail_image': plot_img,
        'plot_image': plot_img,
        'model_path': final_model_path,
        'image_path': str(wsi_path),
        'visualization_time': viz_time,
        'tumor_patches': tumor_patches,
        'normal_patches': normal_patches
    }

    # Send final update to UI
    if progress_callback:
        progress_callback({
            'type': 'complete',
            'result': result
        })

    return result


# Backward compatibility wrapper
def run_pathology_inference(
    wsi_path,
    model_dir="demo/custom_trained/pathology_tumor_detection/models",
    tile_size=224,
    stride=None,
    batch_size=512,
    scale_factor=0.04,
    device_id=0,
    show_plot=True
):
    """
    Wrapper function for backward compatibility.
    Processes pathology images with DataLoader-based batch processing.
    """
    if stride is None:
        stride = tile_size

    output_dir = "demo/output"

    return main(
        wsi_path=wsi_path,
        output_dir=output_dir,
        model_dir=model_dir,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        scale_factor=scale_factor,
        device_id=device_id
    )


if __name__ == "__main__":
    result = main()
    print("\nInference completed successfully!")
    print(f"Results saved to: {result['output_image_path']}")
    print(f"CSV saved to: {result['output_csv_path']}")