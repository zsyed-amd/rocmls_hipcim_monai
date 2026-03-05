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

import os
import time
import cv2
import numpy as np
import cupy as cp
import torch
import matplotlib.pyplot as plt
import csv
from PIL import Image, ImageDraw
from hipcim import CuImage
from pathlib import Path
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import Compose, CastToTyped, ScaleIntensityRanged, ToTensord

def create_pathology_preprocessing():
    """Create preprocessing pipeline for pathology images (same as training)"""
    return Compose([
        CastToTyped(keys="image", dtype=np.float32),
        ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ToTensord(keys="image")
    ])

def create_memory_efficient_thumbnail(img, width, height, scale_factor):
    """Create a thumbnail by sampling tiles across the image to avoid loading full resolution"""
    print("Creating memory-efficient thumbnail...")
    
    target_width = int(width * scale_factor)
    target_height = int(height * scale_factor)
    
    # Calculate sampling parameters
    tiles_x = min(20, target_width // 10)  # Sample at most 20 tiles in X direction
    tiles_y = min(20, target_height // 10)  # Sample at most 20 tiles in Y direction
    
    tile_width = width // tiles_x
    tile_height = height // tiles_y
    
    # Create thumbnail by combining sampled tiles
    thumbnail = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    
    target_tile_width = target_width // tiles_x
    target_tile_height = target_height // tiles_y
    
    for y in range(tiles_y):
        for x in range(tiles_x):
            try:
                # Calculate source coordinates
                src_x = x * tile_width
                src_y = y * tile_height
                
                # Read small tile
                tile = img.read_region(location=(src_x, src_y), 
                                     size=(tile_width, tile_height), level=0)
                tile_np = cp.asnumpy(tile)
                tile_pil = Image.fromarray(tile_np).convert("RGB")
                
                # Resize tile and paste into thumbnail
                tile_resized = tile_pil.resize((target_tile_width, target_tile_height), Image.Resampling.LANCZOS)
                
                # Calculate destination coordinates
                dst_x = x * target_tile_width
                dst_y = y * target_tile_height
                
                thumbnail.paste(tile_resized, (dst_x, dst_y))
                
            except Exception as e:
                print(f"Warning: Failed to sample tile at ({src_x}, src_y): {e}")
                continue
    
    print(f"Memory-efficient thumbnail created: {tiles_x}x{tiles_y} tiles sampled")
    return thumbnail

def main(
    wsi_path="demo/data/pathology_tumor_detection/tumor_001.tif",
    output_dir="demo/output",
    model_path=None,  # Direct path to specific model file (from Streamlit UI)
    model_dir="demo/custom_trained/pathology_tumor_detection/models",  # Fallback directory search
    tile_size=224,
    stride=None,  # Will default to tile_size for no overlap
    batch_size=4,
    scale_factor=0.04,
    device_id=0,  # Default to device 0 (most commonly available)
    confidence_threshold=0.5,  # Confidence threshold for tumor classification
    progress_callback=None  # Callback for streaming progress updates
):
    """
    Memory-efficient pathology inference with batch processing
    """
    # Set stride to tile_size if not specified (no overlap, like bundle script)
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

    # Get image name for CSV
    image_name = Path(wsi_path).stem

    # Load the whole slide image metadata only
    img = CuImage(str(wsi_path))
    height, width = img.shape[:2]
    print(f"Original image shape: {img.shape}")
    print(f"Height: {height} Width: {width}")
    print(f"Image size: {height * width:,} pixels")

    # Create preprocessing transform (similar to torchvision transforms)
    preprocessing = create_pathology_preprocessing()

    # DON'T load full image into memory - this was causing OOM
    print("Processing tiles without loading full image into memory (memory efficient mode)")
    
    # We'll collect tumor coordinates and draw them on a downscaled version later
    tumor_coordinates = []

    # Select appropriate device based on availability
    if torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        if device_id >= available_devices:
            print(f"Warning: Requested device {device_id} not available. Using device 0 instead.")
            device_id = 0
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (available CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0})")

    # Load pathology model
    model_load_start = time.time()

    # Determine which model to use
    if model_path is not None:
        # Use specific model path provided (from Streamlit UI)
        model_path_str = str(model_path)  # Ensure it's a string
        if not os.path.exists(model_path_str):
            raise FileNotFoundError(f"Specified model file not found: {model_path_str}")
        print(f"Using selected model from UI: {model_path_str}")
        final_model_path = model_path_str
    else:
        # Fallback: Look for model files in model_dir
        model_files = []
        model_extensions = ['.pt', '.pth', '.pkl', '.ts']

        for ext in model_extensions:
            model_files.extend(Path(model_dir).glob(f"*{ext}"))

        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        final_model_path = str(model_files[0])
        print(f"Using discovered model: {final_model_path}")

    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=False) #similar to bundle configuration
    # Load checkpoint with proper device mapping to avoid device mismatch errors
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    predictions = []
    tumor_count = 0
    tiles_processed = 0
    tumor_patches = []  # Store tumor patch images for UI display
    normal_patches = []  # Store normal patch images for UI display
    all_recent_patches = []  # Store recent patches for shuffled display

    print(f"Starting tile processing with batch_size={batch_size}...")
    print(f"Configuration: tile_size={tile_size}, stride={stride}")

    # For large tile sizes, reduce batch size to prevent memory issues
    if tile_size >= 2000:
        batch_size = min(batch_size, 4)
        print(f"Large tile size detected, reducing batch_size to {batch_size}")

    # Calculate expected tiles for progress reporting
    expected_tiles_x = (width - tile_size) // stride + 1
    expected_tiles_y = (height - tile_size) // stride + 1
    expected_total_tiles = expected_tiles_x * expected_tiles_y
    print(f"Expected tiles: {expected_total_tiles:,} ({expected_tiles_x} x {expected_tiles_y})")

    inference_start = time.time()

    # Collect tiles for batch processing
    tiles_batch = []
    coords_batch = []

    with torch.no_grad():
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)

                # Skip tiles that are too small (similar to bundle script)
                if tile_width < tile_size or tile_height < tile_size:
                    continue

                # Extract tile using hipcim (like bundle script) - only one tile at a time
                tile = img.read_region(location=(x, y), size=(tile_width, tile_height), level=0)
                tile_np = cp.asnumpy(tile)
                tile_pil = Image.fromarray(tile_np).convert("RGB")

                # Resize tile to model input size (224x224 for pathology bundle)
                tile_resized = tile_pil.resize((224, 224), Image.Resampling.LANCZOS)

                # Convert to numpy array and ensure correct format (like bundle script)
                tile_array = np.array(tile_resized)

                # Convert from HWC to CHW format for MONAI (like bundle script)
                if tile_array.ndim == 3 and tile_array.shape[2] == 3:  # RGB image
                    tile_chw = np.transpose(tile_array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                else:
                    tile_chw = tile_array

                tiles_batch.append(tile_chw)
                coords_batch.append((x, y, tile_width, tile_height))

                # Process batch when full or at end of image
                at_end_of_image = (y + stride >= height) and (x + tile_size >= width)
                if len(tiles_batch) == batch_size or at_end_of_image:
                    # Process current batch
                    batch_tensors = []
                    for tile_chw in tiles_batch:
                        patch_dict = {"image": tile_chw.astype(np.float32)}
                        patch_processed = preprocessing(patch_dict)
                        tensor = patch_processed["image"]
                        batch_tensors.append(tensor)

                    if batch_tensors:
                        batch_tensor = torch.stack(batch_tensors).to(device)
                        batch_start = time.time()
                        outputs = model(batch_tensor)
                        batch_time = time.time() - batch_start

                        # Process outputs (use sigmoid for pathology bundle, like bundle script)
                        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
                        pred_classes = (probabilities > confidence_threshold).astype(int)  # Binary classification with threshold

                        # Store results and collect tumor coordinates for later visualization
                        min_length = min(len(probabilities), len(pred_classes), len(coords_batch))
                        if min_length != len(coords_batch):
                            print(f"Warning: Length mismatch - probabilities: {len(probabilities)}, coords_batch: {len(coords_batch)}")

                        for i in range(min_length):
                            prob, pred_class = probabilities[i], pred_classes[i]
                            coord_x, coord_y, coord_w, coord_h = coords_batch[i]
                            predictions.append(((coord_x, coord_y), pred_class, prob))
                            tiles_processed += 1

                            if pred_class == 1:
                                tumor_count += 1
                                print(f"Tumor detected at ({coord_x}, {coord_y}) - prob: {prob:.4f}")
                                # Store tumor coordinates for later visualization (don't draw yet)
                                tumor_coordinates.append((coord_x, coord_y, coord_w, coord_h))

                                # Store tumor patch for UI display
                                patch_data = {
                                    'image': tiles_batch[i],
                                    'prob': float(prob),
                                    'coords': (coord_x, coord_y),
                                    'label': 'Tumor'
                                }
                                tumor_patches.append(patch_data)
                                all_recent_patches.append(patch_data)

                                # Send immediate update when tumor is detected
                                if progress_callback:
                                    import random
                                    shuffled_patches = all_recent_patches.copy()
                                    random.shuffle(shuffled_patches)
                                    display_patches = shuffled_patches[:10]

                                    elapsed_time = time.time() - inference_start
                                    true_tiles_per_sec = tiles_processed / elapsed_time if elapsed_time > 0 else 0
                                    eta_seconds = (expected_total_tiles - tiles_processed) / true_tiles_per_sec if true_tiles_per_sec > 0 else 0
                                    eta_minutes = eta_seconds / 60

                                    progress_callback({
                                        'type': 'progress',
                                        'tiles_processed': tiles_processed,
                                        'total_tiles': expected_total_tiles,
                                        'tumor_count': tumor_count,
                                        'tiles_per_sec': true_tiles_per_sec,
                                        'eta_minutes': eta_minutes,
                                        'display_patches': display_patches
                                    })
                            else:
                                # Store normal patch for UI display
                                patch_data = {
                                    'image': tiles_batch[i],
                                    'prob': float(prob),
                                    'coords': (coord_x, coord_y),
                                    'label': 'Normal'
                                }
                                normal_patches.append(patch_data)
                                all_recent_patches.append(patch_data)

                            # Keep only recent 50 patches for dynamic shuffling
                            if len(all_recent_patches) > 50:
                                all_recent_patches.pop(0)

                        # Performance reporting with better progress tracking
                        if tiles_processed % (batch_size * 10) == 0:
                            elapsed_time = time.time() - inference_start
                            true_tiles_per_sec = tiles_processed / elapsed_time if elapsed_time > 0 else 0
                            # gpu_tiles_per_sec = len(batch_tensors) / batch_time if batch_time > 0 else 0
                            eta_seconds = (expected_total_tiles - tiles_processed) / true_tiles_per_sec if true_tiles_per_sec > 0 else 0
                            eta_minutes = eta_seconds / 60

                            progress_msg = f"Progress: {tiles_processed}/{expected_total_tiles} tiles ({tiles_processed/expected_total_tiles*100:.1f}%) | Overall: {true_tiles_per_sec:.1f} tiles/s | ETA: {eta_minutes:.1f}min"
                            print(progress_msg)

                            # Send progress update to UI with shuffled patches
                            if progress_callback:
                                import random
                                # Shuffle and select patches to display
                                shuffled_patches = all_recent_patches.copy()
                                random.shuffle(shuffled_patches)
                                display_patches = shuffled_patches[:10]  # Show 10 random recent patches

                                progress_callback({
                                    'type': 'progress',
                                    'tiles_processed': tiles_processed,
                                    'total_tiles': expected_total_tiles,
                                    'tumor_count': tumor_count,
                                    'tiles_per_sec': true_tiles_per_sec,
                                    'eta_minutes': eta_minutes,
                                    'display_patches': display_patches
                                })

                    # Clear batch
                    tiles_batch = []
                    coords_batch = []

    inference_time = time.time() - inference_start

    print(f"Processing complete!")
    print(f"Total tiles processed: {tiles_processed}")
    print(f"Tiles predicted as tumor: {tumor_count}")

    # Calculate tumor percentage
    tumor_percentage = (tumor_count / tiles_processed) * 100 if tiles_processed > 0 else 0
    print(f"Tumor percentage: {tumor_percentage:.1f}%")

    # Overall classification
    overall_classification = 'TUMOR DETECTED' if tumor_percentage > 50 else 'PREDOMINANTLY NORMAL'
    print(f"Overall classification: {overall_classification}")

    # Performance metrics
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Processing speed: {tiles_processed/inference_time:.1f} tiles/sec")

    # Create memory-efficient visualization
    print("Creating memory-efficient annotated image...")
    viz_start = time.time()
    
    try:
        # Try to use pyramid levels for efficient downscaling
        num_levels = img.resolutions['level_count']
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)
        
        # Find appropriate pyramid level
        best_level = min(num_levels - 1, 4)  # Cap at level 4 for memory safety
        level_dims = img.resolutions['level_dimensions'][best_level]
        print(f"Using pyramid level {best_level}: {level_dims} for efficient visualization")
        
        # Read from pyramid level
        downscaled_img = img.read_region(location=(0, 0), size=level_dims, level=best_level)
        img_np = cp.asnumpy(downscaled_img)
        full_image_pil = Image.fromarray(img_np).convert("RGB")
        
        # Resize to exact target dimensions
        full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    except Exception as e:
        print(f"Pyramid method failed ({e}), using tile sampling fallback")
        # Fallback to tile sampling method
        full_image_pil = create_memory_efficient_thumbnail(img, width, height, scale_factor)
    
    # Get actual dimensions for coordinate scaling
    actual_width, actual_height = full_image_pil.size
    actual_scale_x = actual_width / width
    actual_scale_y = actual_height / height
    
    print(f"Output image size: {actual_width}x{actual_height}")
    print(f"Actual scaling factors: X={actual_scale_x:.6f}, Y={actual_scale_y:.6f}")
    
    # Draw tumor annotations with correct coordinate scaling
    draw = ImageDraw.Draw(full_image_pil)
    rectangle_width = max(1, min(5, int(3 / scale_factor)))
    
    annotations_drawn = 0
    for coord_x, coord_y, coord_w, coord_h in tumor_coordinates:
        # Use actual scaling factors for precise coordinate mapping
        scaled_x = int(coord_x * actual_scale_x)
        scaled_y = int(coord_y * actual_scale_y)
        scaled_w = max(1, int(coord_w * actual_scale_x))
        scaled_h = max(1, int(coord_h * actual_scale_y))
        
        # Bounds checking
        if (scaled_x + scaled_w >= actual_width or scaled_y + scaled_h >= actual_height or 
            scaled_x < 0 or scaled_y < 0):
            continue
        
        # Draw red rectangle for tumor detection
        draw.rectangle([scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h],
                      outline="red", width=rectangle_width)
        annotations_drawn += 1
    
    print(f"Annotations drawn: {annotations_drawn}/{len(tumor_coordinates)}")
    viz_time = time.time() - viz_start
    print(f"Visualization created in {viz_time:.2f}s")

    # Convert to numpy for consistent output format
    img_np = np.array(full_image_pil)
    
    # Convert RGB to BGR for OpenCV saving (if needed)
    img_small_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.shape[2] == 3 else img_np

    # Save the annotated image as PNG
    success = cv2.imwrite(str(output_image_path), img_small_rgb)
    if success:
        print(f"Pathology inference image saved successfully at {output_image_path}")
    else:
        print("Failed to save image")

    # Save CSV with predictions
    print(f"Saving prediction results to CSV: {output_csv_path}")
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['image_name', 'x_coordinate', 'y_coordinate', 'prediction_class', 'prediction_probability'])

        # Write prediction data
        for (x, y), pred_class, prob in predictions:
            csv_writer.writerow([image_name, x, y, pred_class, prob])

    print(f"CSV file saved with {len(predictions)} predictions")

    # For Streamlit UI, use RGB format
    plot_img = cv2.cvtColor(img_small_rgb, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_small_rgb

    # Return results
    total_time = time.time() - start_time
    result = {
        'predictions': predictions,
        'tumor_count': tumor_count,
        'total_tiles': tiles_processed,
        'tumor_percentage': tumor_percentage,
        'overall_classification': overall_classification,
        'processing_time': inference_time,
        'total_time': total_time,
        'tiles_per_second': tiles_processed/inference_time if inference_time > 0 else 0,
        'output_image_path': str(output_image_path),
        'output_csv_path': str(output_csv_path),
        'output_image': plot_img,  # Complete annotated image for Streamlit
        'thumbnail_image': plot_img,  # Same complete image
        'plot_image': plot_img,  # Matplotlib plot with annotations
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

# Backward compatibility functions
def run_pathology_inference(
    wsi_path,
    model_dir="demo/custom_trained/pathology_tumor_detection/models",
    tile_size=224,
    stride=None,
    batch_size=4,  # Default batch processing
    scale_factor=0.04,
    device_id=0,  # Default to device 0 (most commonly available)
    show_plot=True
):
    """
    Wrapper function for backward compatibility
    Processes pathology images with batch processing and configurable tile_size/stride
    """
    if stride is None:
        stride = tile_size  # Default: no overlap (like bundle script)

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
    # Example usage - same pattern as pathology_bundle_infer.py
    result = main()
    print("\nInference completed successfully!")
    print(f"Results saved to: {result['output_image_path']}")
    print(f"CSV saved to: {result['output_csv_path']}")