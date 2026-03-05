# AMD ROCm-LS Demo Application

<<<<<<< HEAD
<p align="center">
  <img src="assets/amd-logo.png" alt="AMD Logo" width="200"/>
</p>

**Accelerated Imaging & AI Workflows for Life Sciences**  
**Featuring hipCIM and MONAI on AMD Instinct™ GPUs**
=======
<p align="right">
  <img src="assets/amd-logo.png" alt="AMD Logo" width="200"/>
</p>

**Accelerated Imaging & AI Workflows for Life Sciences - Featuring hipCIM and MONAI on AMD Instinct™ GPUs**
>>>>>>> 70864bf3f62b5317a7e915183b18e4dd0248198d

---

## Overview

This interactive Streamlit application showcases the capabilities of **hipCIM** and **MONAI** — two ROCm-LS-optimized life sciences platforms for AMD Instinct GPUs. The demo features a professional, AMD-branded interface with comprehensive business use cases and technical documentation for each workflow.

### Current Product Coverage

| Product | Description | Status |
|---------|-------------|--------|
| **hipCIM** | GPU-accelerated computational imaging and microscopy | ✅ Available |
| **MONAI** | Medical AI model training and inference | ✅ Training Available |

---

## hipCIM

> ⚠️ **Early Access** — hipCIM is in early access. Production workloads are not recommended.

**hipCIM** (HIP Computational Imaging and Microscopy) is an advanced GPU-accelerated library designed for large-scale biomedical image analysis on AMD Instinct GPUs.

### Business Applications

- **Digital Pathology**: Accelerate whole-slide image analysis for cancer diagnosis
- **Microscopy Research**: Process high-resolution microscopy data in real-time
- **Medical Imaging**: Enable rapid image preprocessing for clinical workflows

### Key Transformations

| Transform | Use Case | Performance |
|-----------|----------|-------------|
| Stain Separation | Isolate H&E stain channels for quantitative pathology | Up to 50x GPU speedup |
| Gabor Filter | Texture analysis for tissue classification | Real-time processing |
| Sobel Edge Detection | Boundary detection for cell segmentation | GPU-accelerated |
| Morphological Ops | Binary dilation, small object removal | Batch processing |
| Geometric Transforms | Rotation, affine warping for registration | Sub-second latency |

---

## MONAI

> ⚠️ **Early Access** — MONAI is in early access. Production workloads are not recommended.

**MONAI** (Medical Open Network for AI) is the leading open-source, PyTorch-based framework for deep learning in healthcare imaging, now optimized for AMD ROCm.

### Supported Models

#### Spleen CT Segmentation
- **Clinical Application**: Automated organ segmentation for surgical planning and trauma assessment
- **Architecture**: 3D U-Net with residual connections
- **Input**: 3D CT volumes (NIfTI format)

#### Pathology Tumor Detection  
- **Clinical Application**: Automated detection of metastatic tissue in lymph node sections
- **Architecture**: DenseNet-121 adapted for histopathology
- **Input**: Whole-slide images (SVS/TIF format)

---

## Quick Start

### Prerequisites

- AMD Instinct™ MI300 GPU or newer
- ROCm 6.4+
- Docker with GPU support
- Git LFS

### Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/ROCm-LS/examples
cd examples

# Install Git LFS and pull large files
git lfs install
git lfs pull

# Build the Docker image
docker build --no-cache -t rocm-ls-demo .

# Run the container
docker run -it -p 8501:8501 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --shm-size=128GB \
  rocm-ls-demo
```

Access the application at: **http://localhost:8501**

### Development Mode (Live Reload)

For development with live code updates:

```bash
docker run -it -p 8501:8501 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v $(pwd):/rocm-ls-examples \
  rocm-ls-demo
```

Changes to Python files will automatically reload the Streamlit app.

---

## Dataset Preparation

### Pathology Tumor Detection

Download the Camelyon dataset from [AWS Open Data Registry](https://registry.opendata.aws/camelyon/):

```bash
# Place files in the correct directories
mkdir -p sample_images/camelyon
mkdir -p demo/data/pathology_tumor_detection

# Copy tumor_001.tif and tumor_101.tif to sample_images/camelyon/
# Copy tumor_001.tif to demo/data/pathology_tumor_detection/
```

### Spleen CT Segmentation

```bash
# Create directory and download dataset
mkdir -p demo/data/spleen_ct_segmentation/

wget -qO- https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar | \
tar -xf - --strip-components=2 --wildcards \
  -C demo/data/spleen_ct_segmentation/ 'Task09_Spleen/imagesTs/spleen_*.nii.gz'
```

### hipCIM Test Data

```bash
./run_amd download_testdata
```

---

## Application Features

### Professional UI with AMD Branding

- **Teal color scheme** (#00A3E0) aligned with AMD brand guidelines
- **Clean, modern layout** with organized sidebars and main panels
- **Contextual help** with business use cases and technical descriptions

### hipCIM Tab

| Feature | Description |
|---------|-------------|
| Image Selection | Load WSI images or upload custom files |
| Tile Extraction | Configure tile size and position |
| Pipeline Builder | Drag-and-drop transformation pipeline |
| Side-by-Side View | Compare Reference, CPU, and GPU results |
| Performance Metrics | Real-time GPU vs CPU speedup graphs |

### MONAI Tab

| Feature | Description |
|---------|-------------|
| Model Zoo | Pre-configured models with metadata |
| Hyperparameter Tuning | Interactive sliders for all parameters |
| Training Dashboard | Real-time loss curves and metrics |
| Result Visualization | Contour overlays and sample predictions |
| Model Export | MONAI bundle format for reproducibility |

---

## Project Structure

```
.
├── rocm-ls_demo.py          # Main application entry point
├── Dockerfile               # Container build configuration
├── supervisord.conf         # Process management
├── requirements.txt         # Python dependencies
│
├── components/              # Shared UI components
│   ├── state.py            # Session state management
│   ├── styles.py           # CSS and theming
│   ├── tooltips.py         # Help text and documentation
│   └── ...
│
├── products/
│   ├── hipcim/             # hipCIM implementation
│   │   ├── layout.py       # Main panel layout
│   │   ├── sidebar.py      # Controls and pipeline builder
│   │   ├── transforms.py   # GPU/CPU transform functions
│   │   └── ...
│   │
│   └── monai/              # MONAI implementation
│       ├── training/       # Training workflows
│       │   ├── models/     # Model definitions (zoo.py)
│       │   └── driver.py   # Training orchestration
│       └── inference/      # Inference workflows (WIP)
│
├── demo/
│   ├── custom_trained/     # Pre-trained model weights
│   └── data/               # Sample datasets
│
├── trained_models/          # Output for trained models
├── use-cases/              # Jupyter notebooks
│   ├── hipcim/
│   └── monai/
│
└── markdown/               # UI text content
```

---

## Extending the Demo

### Adding New hipCIM Transforms

1. Add transform function to `products/hipcim/transforms.py`
2. Register in `OP_CATALOG` in `products/hipcim/sidebar.py`:

```python
"my_transform": {
    "label": "My Transform",
    "business_use": "Clinical application description",
    "technical_desc": "Algorithm details",
    "param_help": {"param1": "Parameter description"},
    "params": {"param1": {"default": 1.0, "min": 0.0, "max": 10.0}}
}
```

### Adding New MONAI Models

1. Create model script in `products/monai/training/models/`
2. Register in `products/monai/training/models/zoo.py`:

```python
"my_model": {
    "name": "My Model",
    "business_context": "Clinical value proposition",
    "clinical_applications": ["App 1", "App 2"],
    "technical_desc": "Architecture details",
    "params": {...}
}
```

---
<<<<<<< HEAD
=======
## Screenshots
<img width="2780" height="1238" alt="image" src="https://github.com/user-attachments/assets/ff06ed8b-ab56-4fd1-9e9b-22c76bca71b6" />
<img width="2797" height="1145" alt="image" src="https://github.com/user-attachments/assets/f84df211-8a74-4f55-a505-401238c729cc" />
>>>>>>> 70864bf3f62b5317a7e915183b18e4dd0248198d

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not detected | Verify ROCm installation: `rocm-smi` |
| Container fails to start | Check device permissions: `ls -la /dev/kfd /dev/dri` |
| LFS files missing | Run `git lfs pull` |
| Port 8501 in use | Use `-p 8502:8501` to map to different port |

---

## References

- [ROCm-LS Documentation](https://rocm.docs.amd.com/projects/rocm-ls/en/latest/index.html)
- [hipCIM GitHub](https://github.com/ROCm-LS/hipCIM)
- [MONAI for ROCm](https://github.com/ROCm-LS/monai)
- [ROCm-LS Organization](https://github.com/ROCm-LS)
<<<<<<< HEAD

---
=======
>>>>>>> 70864bf3f62b5317a7e915183b18e4dd0248198d

---

---

## License

```
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.
```

---

<p align="center">
  <strong>AMD ROCm-LS</strong> — Accelerating Life Sciences Innovation
</p>
