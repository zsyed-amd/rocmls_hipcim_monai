#!/usr/bin/env python3
"""
Optimized Pathology Inference with GPU Acceleration
Uses PyTorch DataLoader for efficient parallel data loading
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import TorchVisionFCModel
import json

# Try to import cucim/cupy, fall back to openslide
try:
    import cupy as cp
    from hipcim import CuImage
    USE_CUCIM = True
    print("✓ Using cuCIM for image loading (GPU-accelerated)")
except ImportError:
    try:
        import openslide
        USE_CUCIM = False
        print("⚠ cuCIM not available, using OpenSlide")
    except ImportError:
        raise ImportError("Neither cuCIM nor OpenSlide is available")


class WSITileDataset(Dataset):
    """Dataset for extracting tiles from Whole Slide Image"""
    
    def __init__(self, wsi_path, tile_size=224, stride=224):
        self.tile_size = tile_size
        self.stride = stride
        
        # Load WSI
        if USE_CUCIM:
            self.img = CuImage(str(wsi_path))
            self.height, self.width = self.img.shape[:2]
        else:
            self.slide = openslide.OpenSlide(str(wsi_path))
            self.width, self.height = self.slide.dimensions
        
        # Calculate tile positions
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
        
        # Extract tile - optimized path
        if USE_CUCIM:
            tile = self.img.read_region(
                location=(x, y),
                size=(self.tile_size, self.tile_size),
                level=0
            )
            tile_np = cp.asnumpy(tile)
            
            # Direct resize if needed (skip PIL conversion for speed)
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
        
        # HWC -> CHW and normalize to [-1, 1] in one step
        if tile_array.ndim == 3 and tile_array.shape[2] == 3:
            tile_chw = np.transpose(tile_array, (2, 0, 1))
        else:
            tile_chw = tile_array
        
        # Normalize: [0, 255] -> [-1, 1]
        tile_normalized = (tile_chw.astype(np.float32) / 127.5) - 1.0  # Slightly faster than /255*2-1
        
        return {
            'image': torch.from_numpy(tile_normalized),
            'coords': (x, y),
            'idx': idx
        }
    
    def close(self):
        if not USE_CUCIM and hasattr(self, 'slide'):
            self.slide.close()


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    coords = [item['coords'] for item in batch]
    indices = [item['idx'] for item in batch]
    return {
        'images': images,
        'coords': coords,
        'indices': indices
    }


def main():
    parser = argparse.ArgumentParser(description='Optimized Pathology Inference')
    parser.add_argument('--image', '-i', required=True, help='Path to WSI image')
    parser.add_argument('--model', '-m', required=True, help='Path to model file')
    parser.add_argument('--output-dir', '-o', default='demo/output', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=224, help='Tile size')
    parser.add_argument('--stride', type=int, default=224, help='Stride')
    parser.add_argument('--batch-size', '-b', type=int, default=512, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=16, help='DataLoader workers (auto-scaled for multi-GPU)')
    parser.add_argument('--device', '-d', type=int, default=0, help='GPU device ID')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--gpu-ids', type=str, default=None, help='GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--scale-factor', type=float, default=0.04, help='Output image scale')
    
    args = parser.parse_args()
    
    # Setup paths
    wsi_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image = output_dir / "optimized_output.png"
    output_csv = output_dir / "optimized_output.csv"
    output_log = output_dir / "optimized_log.json"
    
    # Logging
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'results': {}
    }
    
    start_time = time.time()
    
    # Setup device
    print("\n" + "="*80)
    print("DEVICE CONFIGURATION")
    print("="*80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if args.multi_gpu:
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(num_gpus))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using Multi-GPU: {gpu_ids}")
    else:
        device = torch.device(f'cuda:{args.device}')
        gpu_ids = [args.device]
        print(f"Using Single GPU: {args.device}")
    
    for gid in gpu_ids:
        print(f"  GPU {gid}: {torch.cuda.get_device_name(gid)}")
        mem_gb = torch.cuda.get_device_properties(gid).total_memory / 1024**3
        print(f"    Memory: {mem_gb:.2f} GB")
    
    # Load model
    print("\n" + "="*80)
    print("MODEL LOADING")
    print("="*80)
    
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=False)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Multi-GPU
    if args.multi_gpu and len(gpu_ids) > 1:
        print(f"Wrapping with DataParallel on GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ PyTorch optimizations enabled")
    
    # GPU warm-up
    print("\nWarming up GPU...")
    dummy = torch.randn(args.batch_size, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    torch.cuda.synchronize()
    del dummy
    print("✓ GPU warm-up complete")
    
    # Create dataset and dataloader
    print("\n" + "="*80)
    print("DATA LOADING")
    print("="*80)
    
    dataset = WSITileDataset(wsi_path, tile_size=args.tile_size, stride=args.stride)
    
    # For multi-GPU, increase workers proportionally
    actual_workers = args.num_workers
    if args.multi_gpu and len(gpu_ids) > 1:
        actual_workers = args.num_workers * len(gpu_ids)
        print(f"Multi-GPU detected: Increasing workers from {args.num_workers} to {actual_workers}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=actual_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster CPU->GPU transfer
        prefetch_factor=4 if actual_workers > 0 else None,  # Prefetch more batches
        persistent_workers=True if actual_workers > 0 else False
    )
    
    print(f"DataLoader configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {actual_workers}")
    print(f"  Pin memory: True")
    print(f"  Prefetch factor: {4 if actual_workers > 0 else 'N/A'}")
    
    # Inference
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    
    predictions = []
    tumor_coords = []
    tiles_processed = 0
    tumor_count = 0
    batch_times = []
    
    inference_start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            
            # Move to GPU (non-blocking with pinned memory)
            images = batch['images'].to(device, non_blocking=True)
            
            # Inference
            outputs = model(images)
            
            # Process results
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            pred_classes = (probs > 0.5).astype(int)
            coords = batch['coords']
            
            # Store results
            for i, (prob, pred_class) in enumerate(zip(probs, pred_classes)):
                x, y = coords[i]
                predictions.append((x, y, pred_class, prob))
                tiles_processed += 1
                
                if pred_class == 1:
                    tumor_count += 1
                    tumor_coords.append((x, y, args.tile_size, args.tile_size))
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - inference_start
                tiles_per_sec = tiles_processed / elapsed
                eta_sec = (len(dataset) - tiles_processed) / tiles_per_sec if tiles_per_sec > 0 else 0
                
                print(f"Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Tiles: {tiles_processed}/{len(dataset)} | "
                      f"Speed: {tiles_per_sec:.1f} tiles/s | "
                      f"ETA: {eta_sec/60:.1f}min | "
                      f"Tumors: {tumor_count}")
    
    inference_time = time.time() - inference_start
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total tiles: {tiles_processed}")
    print(f"Tumor tiles: {tumor_count}")
    print(f"Normal tiles: {tiles_processed - tumor_count}")
    print(f"Tumor percentage: {100*tumor_count/tiles_processed:.2f}%")
    print(f"\nInference time: {inference_time:.2f}s")
    print(f"Throughput: {tiles_processed/inference_time:.2f} tiles/s")
    print(f"Avg batch time: {np.mean(batch_times):.3f}s")
    
    # GPU memory
    print(f"\nGPU Memory Usage:")
    for gid in gpu_ids:
        alloc = torch.cuda.memory_allocated(gid) / 1024**3
        cached = torch.cuda.memory_reserved(gid) / 1024**3
        print(f"  GPU {gid}: {alloc:.2f} GB allocated, {cached:.2f} GB cached")
    
    # Save CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'prediction', 'probability'])
        for x, y, pred, prob in predictions:
            writer.writerow([x, y, pred, prob])
    print(f"✓ CSV saved: {output_csv}")
    
    # Create visualization
    print("Creating visualization...")
    width, height = dataset.width, dataset.height
    target_w = int(width * args.scale_factor)
    target_h = int(height * args.scale_factor)
    
    # Load actual WSI thumbnail
    print(f"Loading WSI thumbnail ({target_w}x{target_h})...")
    if USE_CUCIM:
        try:
            # Use pyramid level if available
            num_levels = dataset.img.resolutions['level_count']
            best_level = min(num_levels - 1, 4)
            level_dims = dataset.img.resolutions['level_dimensions'][best_level]
            print(f"Using cuCIM pyramid level {best_level}: {level_dims}")
            
            downscaled = dataset.img.read_region(location=(0, 0), size=level_dims, level=best_level)
            img_np = cp.asnumpy(downscaled)
            img_pil = Image.fromarray(img_np).convert("RGB")
            img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"cuCIM pyramid failed: {e}, using level 0 with downsampling")
            # Sample a smaller region instead of full image
            sample_w = min(width, 4096)
            sample_h = min(height, 4096)
            sample = dataset.img.read_region(location=(0, 0), size=(sample_w, sample_h), level=0)
            img_np = cp.asnumpy(sample)
            img_pil = Image.fromarray(img_np).convert("RGB")
            img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        # OpenSlide thumbnail
        img_pil = dataset.slide.get_thumbnail((target_w, target_h))
        if img_pil.size != (target_w, target_h):
            img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    print(f"✓ Thumbnail loaded: {img_pil.size}")
    draw = ImageDraw.Draw(img_pil)
    
    # Draw tumor annotations
    scale_x = target_w / width
    scale_y = target_h / height
    
    for x, y, w, h in tumor_coords:
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
    
    img_pil.save(output_image)
    print(f"✓ Image saved: {output_image}")
    
    # Save log
    log_data['results'] = {
        'tiles_processed': tiles_processed,
        'tumor_count': tumor_count,
        'inference_time': inference_time,
        'throughput': tiles_processed / inference_time
    }
    
    with open(output_log, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✓ Log saved: {output_log}")
    
    # Cleanup
    dataset.close()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    main()
