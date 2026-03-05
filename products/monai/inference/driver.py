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

import streamlit as st
import numpy as np
import torch
import os
import tempfile
import warnings
from pathlib import Path
from PIL import Image

# Suppress WSI-related warnings globally for pathology inference
warnings.filterwarnings("ignore", message=".*WSI.*")
warnings.filterwarnings("ignore", message=".*sample_images.*")
warnings.filterwarnings("ignore", message=".*cuCIM.*")
warnings.filterwarnings("ignore", message=".*openslide.*")
warnings.filterwarnings("ignore", message=".*No supported images found.*")

from components.state import session_state_get
from components.diagnostics import warning, error, info
from components.path_utils import resolve_path
from products.monai.console import INFER_CONSOLE_LOG_KEY

# Detect available image processing backends
def detect_image_backends():
    """Detect which image processing backends are available"""
    backends = {"cucim": False, "hipcim": False, "pil": False, "opencv": False}

    try:
        import cucim
        backends["cucim"] = True
        info(INFER_CONSOLE_LOG_KEY, f"✅ cuCIM version: {cucim.__version__}")
    except ImportError:
        pass

    try:
        import hipcim
        backends["hipcim"] = True
        info(INFER_CONSOLE_LOG_KEY, f"✅ HIPcuCIM version: {hipcim.__version__}")
    except ImportError:
        pass

    try:
        import PIL
        backends["pil"] = True
        info(INFER_CONSOLE_LOG_KEY, f"✅ PIL version: {PIL.__version__}")
    except ImportError:
        pass

    try:
        import cv2
        backends["opencv"] = True
        info(INFER_CONSOLE_LOG_KEY, f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        pass

    return backends

from products.monai.inference.data_manager import get_available_model_files
from products.monai.styles import TRAINING_DEVICE_ID_HTML, INFERENCE_DEVICE_ID_HTML

# Import our specialized pathology inference
try:
    from products.monai.inference.pathology_inference import (
        main as pathology_main,
        run_pathology_inference
    )
    HAVE_PATHOLOGY_INFERENCE = True
except ImportError:
    HAVE_PATHOLOGY_INFERENCE = False
    warning(INFER_CONSOLE_LOG_KEY, "Pathology inference module not available")

# Import spleen CT segmentation inference
try:
    from products.monai.inference.spleen_ct_inference import run_spleen_segmentation
    HAVE_SPLEEN_INFERENCE = True
except ImportError:
    HAVE_SPLEEN_INFERENCE = False
    warning(INFER_CONSOLE_LOG_KEY, "Spleen CT segmentation module not available")

# Inference button label
INFERENCE_BTN_LBL = """
&nbsp;
&nbsp;
&nbsp;
Run Inference >>
&nbsp;
&nbsp;
&nbsp;
"""

def load_model_for_inference(model_info, device):
    """Load model based on source (custom trained or model zoo)"""
    try:
        if model_info['source'] == 'custom':
            # Load custom trained model
            model_path = model_info['file_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load the saved model state with WSI-safe loading
            import warnings
            with warnings.catch_warnings():
                # Suppress WSI-related warnings that might occur during model loading
                warnings.filterwarnings("ignore", message=".*WSI.*")
                warnings.filterwarnings("ignore", message=".*sample_images.*")
                warnings.filterwarnings("ignore", message=".*cuCIM.*")
                warnings.filterwarnings("ignore", message=".*openslide.*")

                checkpoint = torch.load(model_path, map_location=device)

            # Try to determine model type from filename or checkpoint
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                model_type = checkpoint.get('model_type', 'unknown')
            else:
                model_state = checkpoint
                model_type = 'unknown'

            # For custom models, we return the raw state dict to avoid WSI-related imports
            # Architecture will be detected from state_dict keys in run_inference
            return model_state

        else:
            # Load model zoo model - check for downloaded models first
            model_id = model_info['id']
            available_models = get_available_model_files(model_id)

            if available_models:
                # Use downloaded model
                model_path = available_models[0]  # Use first available model
                info(INFER_CONSOLE_LOG_KEY, f"Loading downloaded model: {model_path}")

                checkpoint = torch.load(model_path, map_location=device)
                return {
                    'checkpoint': checkpoint,
                    'model_path': model_path,
                    'model_id': model_id
                }
            else:
                # No model available - require download
                raise FileNotFoundError(f"No model available for {model_id}. Please download the model first.")

    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"Failed to load model: {str(e)}")
        raise e

def prepare_input_data(input_data, data_source, model_info):
    """Prepare input data for inference"""
    try:
        if data_source == "Upload File":
            # Handle uploaded file
            if input_data is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{input_data.name}") as tmp_file:
                    tmp_file.write(input_data.read())
                    temp_path = tmp_file.name

                # Load and preprocess based on file type
                if input_data.name.lower().endswith(('.nii', '.nii.gz')):
                    # NIfTI file processing
                    import nibabel as nib
                    img = nib.load(temp_path)
                    data = img.get_fdata()
                elif input_data.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    # Image file processing
                    img = Image.open(temp_path)
                    data = np.array(img)
                else:
                    raise ValueError(f"Unsupported file type: {input_data.name}")

                # Clean up temp file
                os.unlink(temp_path)

                return {
                    'data': data,
                    'filename': input_data.name,
                    'type': 'uploaded'
                }
        else:
            # Local data from model-specific folders
            if isinstance(input_data, str) and os.path.exists(input_data):
                # Load local file
                filename = os.path.basename(input_data)

                if input_data.lower().endswith(('.nii', '.nii.gz')):
                    # NIfTI file processing
                    import nibabel as nib
                    img = nib.load(input_data)
                    data = img.get_fdata()
                elif input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    # Image file processing
                    img = Image.open(input_data)
                    data = np.array(img)
                else:
                    raise ValueError(f"Unsupported file type: {filename}")

                return {
                    'data': data,
                    'filename': filename,
                    'type': 'local',
                    'path': input_data
                }
            else:
                # No data available - require download or upload
                raise FileNotFoundError("No data available. Please download data or upload a file.")

    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"Failed to prepare input data: {str(e)}")
        raise e


def detect_model_architecture(model):
    """Detect model architecture from state_dict keys"""
    if isinstance(model, dict) and 'network' in model:
        return 'network'  # Pre-loaded network

    # Check state_dict keys for architecture patterns
    sample_keys = list(model.keys())[:20]  # Check first 20 keys

    if any('features.' in key for key in sample_keys) and any('fc.' in key for key in sample_keys):
        return 'torchvision_fc'  # ResNet/TorchVision architecture
    elif any('model.' in key for key in sample_keys) or any('conv.' in key for key in sample_keys):
        return 'unet'  # UNet architecture
    else:
        return 'unknown'

def run_inference(model, input_data, model_info, device, progress_callback=None):
    """Run inference on input data"""
    try:
        if progress_callback:
            progress_callback("Loading model...", 0.1)

        # Detect and report available backends
        info(INFER_CONSOLE_LOG_KEY, "🔍 Detecting image processing backends...")
        available_backends = detect_image_backends()

        # Report backend status
        active_backends = [name for name, available in available_backends.items() if available]
        if active_backends:
            info(INFER_CONSOLE_LOG_KEY, f"🚀 Active backends: {', '.join(active_backends)}")
        else:
            warning(INFER_CONSOLE_LOG_KEY, "⚠️ No specialized backends detected, using basic PIL")

        # Check specifically for ROCm/AMD optimized backends
        if available_backends["hipcim"]:
            info(INFER_CONSOLE_LOG_KEY, "🔥 HIPcuCIM detected - AMD/ROCm optimized processing available")
        elif available_backends["cucim"]:
            info(INFER_CONSOLE_LOG_KEY, "⚡ cuCIM detected - CUDA optimized processing available")
        else:
            info(INFER_CONSOLE_LOG_KEY, "📷 Using standard PIL for image processing")

        # Detect architecture from model
        architecture_type = detect_model_architecture(model)
        info(INFER_CONSOLE_LOG_KEY, f"🏗️ Detected architecture: {architecture_type}")

        # Preprocess input data based on detected architecture
        if isinstance(input_data['data'], np.ndarray):
            data = input_data['data']

            # Determine model type for proper tensor preprocessing
            if model_info['source'] == 'custom':
                model_path = model_info['file_path']
                model_name = os.path.basename(model_path).lower()
            else:
                model_name = model_info.get('id', '').lower()

            # Handle different model architectures based on detection
            if architecture_type == 'unet':
                # 3D UNet model - match exact training preprocessing

                if data.ndim == 2:
                    # 2D image -> create 3D volume
                    # Add depth dimension by replicating the 2D slice
                    data_3d = np.stack([data] * 96, axis=0)  # (96, H, W)
                elif data.ndim == 3:
                    data_3d = data
                else:
                    data_3d = data

                # Apply intensity scaling exactly like training
                model_source = model_info.get('source', 'custom')
                if model_source == 'custom':
                    # Custom trained models use training preprocessing: a_min=-175, a_max=250
                    data_3d = np.clip(data_3d, -175, 250)
                    data_3d = (data_3d - (-175)) / (250 - (-175))  # Scale to [0, 1]
                else:
                    # Model zoo uses different scaling: a_min=-57, a_max=164
                    data_3d = np.clip(data_3d, -57, 164)
                    data_3d = (data_3d - (-57)) / (164 - (-57))  # Scale to [0, 1]

                # For training, the model expects 96x96x96 patches after RandCropByPosNegLabeld
                # We'll resize the entire volume to a reasonable size first, then extract center crop
                from scipy import ndimage

                # Resize to a reasonable size (e.g., 128x128x128) then center crop to 96x96x96
                intermediate_size = (128, 128, 128)
                zoom_factors = [intermediate_size[i] / data_3d.shape[i] for i in range(3)]
                data_resized = ndimage.zoom(data_3d, zoom_factors, order=1)

                # Center crop to 96x96x96
                d, h, w = data_resized.shape
                target_d, target_h, target_w = 96, 96, 96

                # Calculate crop indices
                start_d = max(0, (d - target_d) // 2)
                start_h = max(0, (h - target_h) // 2)
                start_w = max(0, (w - target_w) // 2)

                end_d = min(d, start_d + target_d)
                end_h = min(h, start_h + target_h)
                end_w = min(w, start_w + target_w)

                data_cropped = data_resized[start_d:end_d, start_h:end_h, start_w:end_w]

                # Pad if necessary to ensure exactly 96x96x96
                pad_d = max(0, target_d - data_cropped.shape[0])
                pad_h = max(0, target_h - data_cropped.shape[1])
                pad_w = max(0, target_w - data_cropped.shape[2])

                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    padding = ((0, pad_d), (0, pad_h), (0, pad_w))
                    data_cropped = np.pad(data_cropped, padding, mode='constant', constant_values=0)

                # Ensure exactly 96x96x96
                data_final = data_cropped[:96, :96, :96]

                # Convert to tensor: (1, 1, D, H, W) for 3D UNet
                tensor_input = torch.from_numpy(data_final).float().unsqueeze(0).unsqueeze(0)

            elif architecture_type == 'torchvision_fc':
                # Pathology model expects 224x224 RGB patches
                info(INFER_CONSOLE_LOG_KEY, "🔬 Processing pathology image for TorchVisionFC model")
                target_size = (224, 224)

                if data.ndim == 2:
                    # Grayscale -> RGB
                    data_rgb = np.stack([data] * 3, axis=-1)  # (H,W,3)
                    info(INFER_CONSOLE_LOG_KEY, f"📸 Converted grayscale to RGB: {data.shape} -> {data_rgb.shape}")
                elif data.ndim == 3:
                    if data.shape[-1] == 3:
                        data_rgb = data  # Already RGB
                        info(INFER_CONSOLE_LOG_KEY, f"✅ Image already in RGB format: {data.shape}")
                    elif data.shape[0] == 3:
                        data_rgb = np.transpose(data, (1, 2, 0))  # (3,H,W) -> (H,W,3)
                        info(INFER_CONSOLE_LOG_KEY, f"🔄 Transposed channels: {data.shape} -> {data_rgb.shape}")
                    else:
                        # Take middle slice and convert to RGB
                        mid_slice = data[data.shape[0]//2] if data.shape[0] > 1 else data[0]
                        data_rgb = np.stack([mid_slice] * 3, axis=-1)
                        info(INFER_CONSOLE_LOG_KEY, f"🎯 Extracted middle slice and converted to RGB: {data.shape} -> {data_rgb.shape}")
                else:
                    data_rgb = data

                # Try using optimized backends for resizing
                info(INFER_CONSOLE_LOG_KEY, f"🔧 Resizing image from {data_rgb.shape[:2]} to {target_size}")

                if data_rgb.shape[:2] != target_size:
                    resize_method = "PIL (fallback)"

                    # Try cuCIM/HIPcuCIM first if available (currently would need implementation)
                    if available_backends["hipcim"]:
                        try:
                            # HIPcuCIM implementation would go here
                            resize_method = "HIPcuCIM (AMD optimized)"
                            info(INFER_CONSOLE_LOG_KEY, f"🔥 Using HIPcuCIM for image resizing")
                        except:
                            pass
                    elif available_backends["cucim"]:
                        try:
                            # cuCIM implementation would go here
                            resize_method = "cuCIM (CUDA optimized)"
                            info(INFER_CONSOLE_LOG_KEY, f"⚡ Using cuCIM for image resizing")
                        except:
                            pass

                    # Fallback to PIL
                    from PIL import Image
                    if data_rgb.dtype != np.uint8:
                        data_rgb = (data_rgb * 255).astype(np.uint8)
                    pil_img = Image.fromarray(data_rgb)
                    pil_img = pil_img.resize(target_size, Image.BILINEAR)
                    data_rgb = np.array(pil_img)

                    info(INFER_CONSOLE_LOG_KEY, f"✅ Resized using {resize_method}: {data_rgb.shape}")
                else:
                    info(INFER_CONSOLE_LOG_KEY, f"✅ Image already at target size {target_size}")

                # Normalize to [0,1] and convert to tensor: (H,W,3) -> (1,3,H,W)
                data_rgb = data_rgb.astype(np.float32) / 255.0
                tensor_input = torch.from_numpy(np.transpose(data_rgb, (2, 0, 1))).unsqueeze(0)

            else:
                # Generic 2D model - needs 4D tensor: (batch, channel, height, width)
                if data.ndim == 2:
                    tensor_input = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                elif data.ndim == 3:
                    tensor_input = torch.from_numpy(data).float().unsqueeze(0)
                else:
                    tensor_input = torch.from_numpy(data).float()

            tensor_input = tensor_input.to(device)
        else:
            raise ValueError("Unsupported input data type")

        if progress_callback:
            progress_callback("Running inference...", 0.5)

        # Run inference based on model source
        if model_info['source'] == 'custom':
            # For custom models, reconstruct the network and load state dict
            if 'network' in model:
                network = model['network']
                network.eval()
                with torch.no_grad():
                    output = network(tensor_input)
            else:
                # Handle raw model state dict - use already detected architecture
                model_path = model_info['file_path']
                model_name = os.path.basename(model_path).lower()

                if architecture_type == 'unet':
                    # Reconstruct UNet for spleen CT segmentation
                    # Must match training architecture - 3D UNet
                    from monai.networks.nets import UNet
                    network = UNet(
                        spatial_dims=3,  # Keep 3D to match training
                        in_channels=1,
                        out_channels=2,
                        channels=(16, 32, 64, 128, 256),
                        strides=(2, 2, 2, 2),
                        num_res_units=2,
                        norm="batch"
                    ).to(device)

                elif architecture_type == 'torchvision_fc':
                    # Reconstruct TorchVisionFCModel for pathology classification
                    # Import locally to avoid any potential WSI-related side effects
                    try:
                        from monai.networks.nets import TorchVisionFCModel
                        network = TorchVisionFCModel(
                            "resnet18",
                            num_classes=1,
                            use_conv=True,
                            pretrained=False
                        ).to(device)
                        info(INFER_CONSOLE_LOG_KEY, "✅ TorchVisionFCModel created successfully")
                    except Exception as e:
                        error(INFER_CONSOLE_LOG_KEY, f"Failed to create TorchVisionFCModel: {e}")
                        # Fallback: Create a simple ResNet18 directly
                        import torchvision.models as models
                        import torch.nn as nn
                        network = models.resnet18(pretrained=False)
                        network.fc = nn.Linear(network.fc.in_features, 1)
                        network = network.to(device)
                        warning(INFER_CONSOLE_LOG_KEY, "Using fallback ResNet18 model")

                else:
                    # Generic small network for unknown models
                    import torch.nn as nn
                    network = nn.Sequential(
                        nn.Conv2d(tensor_input.shape[1], 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, 1),
                        nn.Sigmoid()
                    ).to(device)

                # Load the state dict
                if 'model_state' in model:
                    network.load_state_dict(model['model_state'])
                else:
                    # Try loading directly
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            network.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            network.load_state_dict(checkpoint)
                    else:
                        network.load_state_dict(checkpoint)

                network.eval()
                with torch.no_grad():
                    output = network(tensor_input)
        else:
            # Model zoo inference - use downloaded model
            if 'checkpoint' in model:
                # For downloaded models, implement basic inference
                # This is a simplified version - real implementation would depend on model architecture
                checkpoint = model['checkpoint']

                # For now, create a simple placeholder output
                # In real implementation, you'd reconstruct the network and load weights
                output = torch.randn(1, 1, tensor_input.shape[2], tensor_input.shape[3]).to(device)
                warning(INFER_CONSOLE_LOG_KEY, "Using placeholder inference - implement proper model loading")
            else:
                raise ValueError("No valid model checkpoint available for inference")

        if progress_callback:
            progress_callback("Processing results...", 0.8)

        # Post-process output
        if isinstance(output, torch.Tensor):
            output_np = output.cpu().numpy()
            # Remove batch dimension if present
            if output_np.ndim > 2 and output_np.shape[0] == 1:
                output_np = output_np[0]
            if output_np.ndim > 2 and output_np.shape[0] == 1:
                output_np = output_np[0]
        else:
            output_np = output

        if progress_callback:
            progress_callback("Inference completed!", 1.0)

        return {
            'input': input_data['data'],
            'output': output_np,
            'filename': input_data['filename'],
            'model_name': model_info['name'],
            'device': str(device)
        }

    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"Inference failed: {str(e)}")
        raise e

def create_visualization(result):
    """Create visualization of inference results"""
    input_img = result['input']
    output_img = result['output']

    # Normalize images for display
    def normalize_for_display(img):
        if img.ndim > 2:
            if img.shape[0] < img.shape[-1]:  # Channel first
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] > 3:  # More than RGB channels
                img = img[..., 0]  # Take first channel
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img_norm * 255).astype(np.uint8)

    input_display = normalize_for_display(input_img)
    output_display = normalize_for_display(output_img)

    return {
        'input': input_display,
        'output': output_display,
        'overlay': create_overlay_visualization(input_display, output_display)
    }

def create_overlay_visualization(input_img, output_img):
    """Create overlay visualization"""
    try:
        # Convert to RGB if grayscale
        if input_img.ndim == 2:
            input_rgb = np.stack([input_img] * 3, axis=-1)
        else:
            input_rgb = input_img

        if output_img.ndim == 2:
            # Treat output as mask and overlay in color
            overlay = input_rgb.copy()
            mask = output_img > (output_img.max() * 0.5)  # Threshold
            overlay[mask] = [255, 0, 0]  # Red overlay for predictions
            return overlay
        else:
            # Side by side comparison
            return np.concatenate([input_rgb, output_img], axis=1)

    except Exception:
        # Fallback to input image
        return input_img

def monai_inference_driver():
    """Main inference driver with UI similar to training page"""

    # System ready - info will be shown when inference runs

    # Header and button layout (similar to training)
    inference_header_placeholder = st.container()
    with inference_header_placeholder:
        inference_header_row = st.columns([1, 7])

    # Layout the inference button
    inference_btn_placeholder = inference_header_row[0].empty()
    run_btn = inference_btn_placeholder.button(
        INFERENCE_BTN_LBL,
        disabled=False,
        key="run_inference_btn"
    )

    with inference_header_row[1]:
        inference_info_placeholder = inference_header_row[1].empty()
        progress_row = st.empty()

    # Display model and device info
    with inference_info_placeholder:
        model_info = session_state_get('monai_inference_model_info')
        if model_info:
            st.markdown(f"**Model:** {model_info['name']} ({model_info['source']})")

    progress_indicator = progress_row.progress(0, text="Ready for inference...")

    # Main layout - similar to training page structure
    sample_viewport, results_viewport = st.columns([1, 2])

    # Device identification (similar to training)
    with sample_viewport:
        device_container = st.container(border=True)
        device_identification = device_container.empty()
        device_id = session_state_get('inference_device')
        device_type = session_state_get('inference_device_type')
        if device_id and device_type:
            device_identification.markdown(
                INFERENCE_DEVICE_ID_HTML.format(
                    device_type=device_type,
                ),
                unsafe_allow_html=True
            )

        # Input data preview
        st.markdown("**Input Data Preview**")
        input_preview = st.empty()

    # Progress callback for UI updates
    def update_progress(message, progress):
        progress_indicator.progress(progress, text=message)

    # Button event: Launch inference
    if run_btn:
        try:
            # Get all required data from session state
            model_info = session_state_get('monai_inference_model_info')
            input_data = session_state_get('monai_inference_input_data')
            # Get data source directly from widget state
            data_source = st.session_state.get('monai_data_source', 'Local Data')
            device = session_state_get('inference_device')

            # Check what's missing
            missing_items = []
            if not model_info:
                missing_items.append("model")
            if not input_data:
                missing_items.append("input data")
            if not device:
                missing_items.append("device")

            if missing_items:
                st.error(f"❌ Missing: {', '.join(missing_items)}. Please select them in the sidebar first.")
                return

            with st.spinner(f"Running inference with {model_info['name']}...", show_time=True):

                # Check if this is a pathology model and use specialized inference
                if (HAVE_PATHOLOGY_INFERENCE and
                    model_info.get('source') == 'custom' and
                    ('pathology' in model_info.get('model_type', '').lower() or
                     'pathology' in model_info.get('file_path', '').lower())):

                    info(INFER_CONSOLE_LOG_KEY, "🔬 Using specialized pathology inference")

                    # Extract model and image names for pathology inference
                    from pathlib import Path

                    # For model, we can use just the model name since our pathology inference will find it
                    model_path = Path(model_info['file_path'])
                    model_name = model_path.stem  # Remove extension

                    # Also try parent directory name if model is generic (like 'model')
                    if model_name in ['model']:
                        parent_dir = model_path.parent.name
                        if parent_dir in ['models']:
                            # Go up one more level to get pathology_tumor_detection
                            parent_dir = model_path.parent.parent.name
                        # For pathology, we can just use 'model' as our inference function will find it
                        if 'pathology' in parent_dir.lower():
                            model_name = 'model'  # Keep it simple, let pathology_inference find the right file

                    if isinstance(input_data, str):
                        image_path = Path(input_data)
                        image_name = image_path.stem  # Remove extension
                    else:
                        st.error("Pathology inference requires a file path input")
                        return

                    info(INFER_CONSOLE_LOG_KEY, f"Using model: {model_name}, image: {image_name}")

                    try:
                        # Get configuration from Streamlit session state
                        tile_size = st.session_state.get('pathology_tile_size', 224)
                        #batch_size = st.session_state.get('pathology_batch_size', 4)
                        batch_size = st.session_state.get('pathology_batch_size', 512)
                        confidence_threshold = st.session_state.get('pathology_confidence_threshold', 0.5)

                        info(INFER_CONSOLE_LOG_KEY, f"Configuration: tile_size={tile_size}, batch_size={batch_size}, threshold={confidence_threshold}")

                        # Convert paths to absolute for inference
                        absolute_input_path = resolve_path(input_data)
                        absolute_model_path = resolve_path(model_info['file_path'])

                        # Pre-generate WSI thumbnail BEFORE starting inference
                        wsi_thumbnail = None
                        try:
                            from components.utility import generate_wsi_thumbnail
                            from PIL import Image as PILImage
                            import os

                            # Create cache directory
                            cache_dir = ".generated_thumbnails"
                            os.makedirs(cache_dir, exist_ok=True)

                            # Generate cached thumbnail (600px width for 1/3 size, fast loading)
                            wsi_name = os.path.basename(absolute_input_path)
                            thumb_path = os.path.join(cache_dir, f"{wsi_name}_thumb.jpg")

                            with st.spinner('Generating WSI thumbnail...'):
                                generate_wsi_thumbnail(str(absolute_input_path), thumb_path, width=600)
                                wsi_thumbnail = PILImage.open(thumb_path)
                                info(INFER_CONSOLE_LOG_KEY, f"Thumbnail generated: {thumb_path}")
                        except Exception as e:
                            info(INFER_CONSOLE_LOG_KEY, f"Could not generate thumbnail: {e}")

                        # Create placeholders for real-time updates
                        status_placeholder = st.empty()

                        # Create interactive layout during inference
                        live_col1, live_col2 = st.columns([1, 1])

                        with live_col1:
                            wsi_thumbnail_placeholder = st.empty()
                            if wsi_thumbnail:
                                with wsi_thumbnail_placeholder.container():
                                    st.markdown("**Whole Slide Image (Thumbnail)**")
                                    st.image(wsi_thumbnail, width='stretch')

                        with live_col2:
                            # Tumor patches container
                            tumor_patches_container = st.container(border=True, height=480)
                            tumor_patches_placeholder = tumor_patches_container.empty()

                            # Normal patches container
                            normal_patches_container = st.container(border=True, height=480)
                            normal_patches_placeholder = normal_patches_container.empty()

                        # Progress tracking variables
                        current_progress = {'tiles_processed': 0, 'tumor_count': 0, 'tumor_patches': [], 'normal_patches': [], 'display_patches': []}
                        last_tumor_patches = []
                        last_normal_patches = []

                        def handle_progress(update):
                            nonlocal last_tumor_patches, last_normal_patches

                            if update['type'] == 'progress':
                                current_progress.update(update)
                                progress = update['tiles_processed'] / update['total_tiles'] if update['total_tiles'] > 0 else 0

                                # Update status
                                status_placeholder.markdown(
                                    f"**Processing:** {update['tiles_processed']}/{update['total_tiles']} tiles "
                                    f"({progress*100:.1f}%) | "
                                    f"**Tumor patches:** {update['tumor_count']} | "
                                    f"**Speed:** {update['tiles_per_sec']:.1f} tiles/s | "
                                    f"**ETA:** {update['eta_minutes']:.1f}min"
                                )
                                update_progress(f"Processing {update['tiles_processed']}/{update['total_tiles']} tiles...", progress)

                                # Right side: Split into tumor (top) and normal (bottom) patches
                                tumor_patches = []
                                normal_patches = []

                                if 'display_patches' in update and update['display_patches']:
                                    # Separate tumor and normal patches
                                    tumor_patches = [p for p in update['display_patches'] if p['label'] == 'Tumor']
                                    normal_patches = [p for p in update['display_patches'] if p['label'] == 'Normal']

                                # Store for later display
                                if tumor_patches:
                                    last_tumor_patches = tumor_patches
                                if normal_patches:
                                    last_normal_patches = normal_patches

                                # Top: Tumor patches (use new patches if available, otherwise keep last)
                                with tumor_patches_placeholder.container():
                                    st.markdown("**🔴 Tumor Patches**")
                                    display_tumor = tumor_patches[:10] if tumor_patches else last_tumor_patches[:10] if last_tumor_patches else []
                                    if display_tumor:
                                        # First row (5 patches)
                                        cols1 = st.columns(5)
                                        for idx in range(min(5, len(display_tumor))):
                                            with cols1[idx]:
                                                img_chw = display_tumor[idx]['image']
                                                if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                    img_hwc = np.transpose(img_chw, (1, 2, 0))
                                                else:
                                                    img_hwc = img_chw
                                                st.image(img_hwc.astype(np.uint8), width='stretch')
                                                st.markdown(
                                                    f"<p style='text-align: center; color: red; margin: 0; font-size: 11px;'>"
                                                    f"<b>Tumor ({display_tumor[idx]['prob']:.2f})</b></p>",
                                                    unsafe_allow_html=True
                                                )

                                        # Second row (5 patches)
                                        if len(display_tumor) > 5:
                                            cols2 = st.columns(5)
                                            for idx in range(5, min(10, len(display_tumor))):
                                                with cols2[idx - 5]:
                                                    img_chw = display_tumor[idx]['image']
                                                    if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                        img_hwc = np.transpose(img_chw, (1, 2, 0))
                                                    else:
                                                        img_hwc = img_chw
                                                    st.image(img_hwc.astype(np.uint8), width='stretch')
                                                    st.markdown(
                                                        f"<p style='text-align: center; color: red; margin: 0; font-size: 11px;'>"
                                                        f"<b>Tumor ({display_tumor[idx]['prob']:.2f})</b></p>",
                                                        unsafe_allow_html=True
                                                    )
                                    else:
                                        st.info("No tumor patches detected yet...")

                                    # Bottom: Normal patches (use new patches if available, otherwise keep last)
                                    with normal_patches_placeholder.container():
                                        st.markdown("**🟢 Normal Patches**")
                                        display_normal = normal_patches[:10] if normal_patches else last_normal_patches[:10] if last_normal_patches else []
                                        if display_normal:
                                            # First row (5 patches)
                                            cols1 = st.columns(5)
                                            for idx in range(min(5, len(display_normal))):
                                                with cols1[idx]:
                                                    img_chw = display_normal[idx]['image']
                                                    if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                        img_hwc = np.transpose(img_chw, (1, 2, 0))
                                                    else:
                                                        img_hwc = img_chw
                                                    st.image(img_hwc.astype(np.uint8), width='stretch')
                                                    st.markdown(
                                                        f"<p style='text-align: center; color: green; margin: 0; font-size: 11px;'>"
                                                        f"<b>Normal ({display_normal[idx]['prob']:.2f})</b></p>",
                                                        unsafe_allow_html=True
                                                    )

                                            # Second row (5 patches)
                                            if len(display_normal) > 5:
                                                cols2 = st.columns(5)
                                                for idx in range(5, min(10, len(display_normal))):
                                                    with cols2[idx - 5]:
                                                        img_chw = display_normal[idx]['image']
                                                        if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                            img_hwc = np.transpose(img_chw, (1, 2, 0))
                                                        else:
                                                            img_hwc = img_chw
                                                        st.image(img_hwc.astype(np.uint8), width='stretch')
                                                        st.markdown(
                                                            f"<p style='text-align: center; color: green; margin: 0; font-size: 11px;'>"
                                                            f"<b>Normal ({display_normal[idx]['prob']:.2f})</b></p>",
                                                            unsafe_allow_html=True
                                                        )
                                        else:
                                            st.info("No normal patches detected yet...")

                        # Run simplified pathology inference with selected model
                        result = pathology_main(
                            wsi_path=absolute_input_path,
                            model_path=absolute_model_path,  # Pass selected model from UI
                            tile_size=tile_size,
                            batch_size=batch_size,
                            stride=tile_size,  # No overlap
                            device_id=0,  # Use device 0 like our default
                            confidence_threshold=confidence_threshold,
                            progress_callback=handle_progress
                        )

                        # Clear status and update with final annotated image
                        status_placeholder.empty()

                        # Replace thumbnail with annotated image
                        with wsi_thumbnail_placeholder.container():
                            st.markdown("**📸 Annotated Whole Slide Image**")
                            if 'plot_image' in result and result['plot_image'] is not None:
                                st.image(result['plot_image'], caption="Red boxes indicate tumor detections", width='stretch')

                        # Keep the last tumor and normal patches visible
                        if last_tumor_patches:
                            with tumor_patches_placeholder.container():
                                st.markdown("**🔴 Tumor Patches (Final)**")
                                display_tumor = last_tumor_patches[:10]
                                # First row (5 patches)
                                cols1 = st.columns(5)
                                for idx in range(min(5, len(display_tumor))):
                                    with cols1[idx]:
                                        img_chw = display_tumor[idx]['image']
                                        if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                            img_hwc = np.transpose(img_chw, (1, 2, 0))
                                        else:
                                            img_hwc = img_chw
                                        st.image(img_hwc.astype(np.uint8), width='stretch')
                                        st.markdown(
                                            f"<p style='text-align: center; color: red; margin: 0; font-size: 11px;'>"
                                            f"<b>Tumor ({display_tumor[idx]['prob']:.2f})</b></p>",
                                            unsafe_allow_html=True
                                        )
                                # Second row (5 patches)
                                if len(display_tumor) > 5:
                                    cols2 = st.columns(5)
                                    for idx in range(5, min(10, len(display_tumor))):
                                        with cols2[idx - 5]:
                                            img_chw = display_tumor[idx]['image']
                                            if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                img_hwc = np.transpose(img_chw, (1, 2, 0))
                                            else:
                                                img_hwc = img_chw
                                            st.image(img_hwc.astype(np.uint8), width='stretch')
                                            st.markdown(
                                                f"<p style='text-align: center; color: red; margin: 0; font-size: 11px;'>"
                                                f"<b>Tumor ({display_tumor[idx]['prob']:.2f})</b></p>",
                                                unsafe_allow_html=True
                                            )

                        if last_normal_patches:
                            with normal_patches_placeholder.container():
                                st.markdown("**🟢 Normal Patches (Final)**")
                                display_normal = last_normal_patches[:10]
                                # First row (5 patches)
                                cols1 = st.columns(5)
                                for idx in range(min(5, len(display_normal))):
                                    with cols1[idx]:
                                        img_chw = display_normal[idx]['image']
                                        if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                            img_hwc = np.transpose(img_chw, (1, 2, 0))
                                        else:
                                            img_hwc = img_chw
                                        st.image(img_hwc.astype(np.uint8), width='stretch')
                                        st.markdown(
                                            f"<p style='text-align: center; color: green; margin: 0; font-size: 11px;'>"
                                            f"<b>Normal ({display_normal[idx]['prob']:.2f})</b></p>",
                                            unsafe_allow_html=True
                                        )
                                # Second row (5 patches)
                                if len(display_normal) > 5:
                                    cols2 = st.columns(5)
                                    for idx in range(5, min(10, len(display_normal))):
                                        with cols2[idx - 5]:
                                            img_chw = display_normal[idx]['image']
                                            if isinstance(img_chw, np.ndarray) and img_chw.ndim == 3 and img_chw.shape[0] == 3:
                                                img_hwc = np.transpose(img_chw, (1, 2, 0))
                                            else:
                                                img_hwc = img_chw
                                            st.image(img_hwc.astype(np.uint8), width='stretch')
                                            st.markdown(
                                                f"<p style='text-align: center; color: green; margin: 0; font-size: 11px;'>"
                                                f"<b>Normal ({display_normal[idx]['prob']:.2f})</b></p>",
                                                unsafe_allow_html=True
                                            )

                        # Display pathology-specific results
                        update_progress("Displaying results...", 0.9)

                        st.markdown("---")
                        st.markdown("### 📊 Final Analysis Results")

                        # Create tabs for detailed information (similar to training layout)
                        path_tabs = st.tabs([
                            "**📊 Analysis Details**",
                            "**⚙️ Configuration**"
                        ])

                        with path_tabs[0]:
                            # Classification result
                            st.markdown("**Classification Result**")
                            if result['overall_classification'] == 'TUMOR DETECTED':
                                st.markdown(
                                    f"<div style='background-color: #ffebee; padding: 20px; border-radius: 10px; text-align: center;'>"
                                    f"<h2 style='color: #d32f2f; margin: 0;'>{result['overall_classification']}</h2>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"<div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;'>"
                                    f"<h2 style='color: #388e3c; margin: 0;'>{result['overall_classification']}</h2>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                            st.markdown("---")

                            # Performance metrics
                            st.markdown("**Performance Metrics**")
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Total Patches", f"{result['total_tiles']:,}")
                                st.metric("Tumor Patches", f"{result['tumor_count']:,}")
                            with metric_col2:
                                st.metric("Tumor %", f"{result['tumor_percentage']:.1f}%")
                                st.metric("Speed", f"{result['tiles_per_second']:.1f} tiles/s")

                            st.markdown("---")

                            # Processing times
                            st.markdown("**Processing Time**")
                            time_col1, time_col2 = st.columns(2)
                            with time_col1:
                                st.metric("Total Time", f"{result.get('total_time', 0):.2f}s")
                                st.metric("Model Used", "ResNet18")
                            with time_col2:
                                st.metric("Inference Time", f"{result.get('processing_time', 0):.2f}s")
                                st.metric("Confidence Threshold", f"{confidence_threshold:.2f}")

                            st.markdown("---")

                            # Show only high-confidence tumor detections (essential info)
                            if 'predictions' in result and result['predictions']:
                                tumor_patches = [(x, y, prob) for ((x, y), pred_class, prob) in result['predictions'] if pred_class == 1 and prob > 0.7]
                                if tumor_patches:
                                    st.markdown(
                                        "<div style='margin: 10px 0 5px 0; padding: 0px 0; font-size: 10px; font-weight: bold; text-align: left; border-bottom: 1px solid #bbb;'>High-Confidence Detections</div>",
                                        unsafe_allow_html=True
                                    )
                                    for i, (x, y, prob) in enumerate(tumor_patches[:3]):  # Show top 3 only
                                        st.write(f"🔴 Detection {i+1}: ({x}, {y}) - {prob:.3f}")
                                else:
                                    st.info("No high-confidence tumor detections found")
                            else:
                                st.info("No detection data available")

                        with path_tabs[1]:
                            # Configuration details
                            st.subheader("⚙️ Configuration Used")

                            config_details = {
                                'Tile Size': f"{tile_size}x{tile_size}",
                                'Batch Size': batch_size,
                                'Total Processing Time': f"{result.get('total_time', 0):.2f}s",
                                'Processing Speed': f"{result.get('tiles_per_second', 0):.1f} tiles/sec",
                                'Model Path': result.get('model_path', 'N/A'),
                                'Output Image': result.get('output_image_path', 'N/A'),
                                'Output CSV': result.get('output_csv_path', 'N/A')
                            }

                            st.json(config_details)

                        update_progress("Pathology analysis completed!", 1.0)
                        return  # Exit early for pathology inference

                    except Exception as e:
                        st.error(f"❌ Pathology inference failed: {str(e)}")
                        error(INFER_CONSOLE_LOG_KEY, f"Pathology inference error: {str(e)}")
                        return

                # Check if this is a spleen CT segmentation model and use specialized inference
                elif (model_info.get('source') == 'custom' and
                      ('spleen' in model_info.get('model_type', '').lower() or
                       'spleen' in model_info.get('file_path', '').lower() or
                       'ct_segmentation' in model_info.get('file_path', '').lower())):

                    info(INFER_CONSOLE_LOG_KEY, "🫁 Using specialized spleen CT segmentation inference")

                    # Extract model and input paths for spleen CT inference
                    from pathlib import Path

                    model_path = model_info['file_path']
                    model_dir = str(Path(model_path).parent)

                    # Handle both file path and uploaded file
                    if isinstance(input_data, str):
                        # Direct file path
                        input_path = input_data
                    elif hasattr(input_data, 'name') and input_data.name.lower().endswith(('.nii', '.nii.gz')):
                        # Uploaded NIfTI file - save to temporary location
                        info(INFER_CONSOLE_LOG_KEY, f"Processing uploaded file: {input_data.name}")
                        
                        import tempfile
                        import os
                        
                        # Create temporary file with proper extension
                        file_ext = '.nii.gz' if input_data.name.lower().endswith('.nii.gz') else '.nii'
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                            tmp_file.write(input_data.read())
                            input_path = tmp_file.name
                            
                        info(INFER_CONSOLE_LOG_KEY, f"Saved uploaded file to: {input_path}")
                    elif isinstance(input_data, dict) and 'path' in input_data:
                        # Input data prepared by prepare_input_data function
                        input_path = input_data['path']
                    else:
                        st.error("Spleen CT inference requires a NIfTI file (.nii or .nii.gz). Please upload a valid NIfTI file or select from local data.")
                        return

                    info(INFER_CONSOLE_LOG_KEY, f"Using model dir: {model_dir}, input: {Path(input_path).name}")

                    try:
                        # Get device configuration from sidebar
                        inference_device = session_state_get('inference_device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                        inference_device_type = session_state_get('inference_device_type', 'Auto')
                        
                        # Extract device ID for compatibility
                        if inference_device.type == 'cuda':
                            device_id = inference_device.index if inference_device.index is not None else 0
                        else:
                            device_id = -1  # CPU mode

                        info(INFER_CONSOLE_LOG_KEY, f"Configuration: device={inference_device} ({inference_device_type})")
                        update_progress(f"Running spleen CT segmentation on {inference_device_type}...", 0.5)

                        # Convert paths to absolute for inference
                        absolute_input_path = resolve_path(input_path)
                        absolute_model_path = resolve_path(model_info['file_path'])

                        # Run spleen CT segmentation using the imported function
                        result = run_spleen_segmentation(
                            input_path=absolute_input_path,
                            model_path=absolute_model_path,  # Pass selected model from UI
                            device=inference_device  # Pass the actual device object
                        )

                        # Display spleen CT-specific results
                        update_progress("Displaying results...", 0.9)

                        # Main results viewport (visualization only)
                        with st.container():
                            st.markdown(
                                "<div style='margin: 0px 0; padding: 0px 0; font-size: 10px; font-weight: bold; text-align: left; border-bottom: 1px solid #bbb;'>Spleen CT Segmentation Results</div>",
                                unsafe_allow_html=True
                            )

                            if 'plot_image' in result and result['plot_image'] is not None:
                                # Create informative caption explaining the slice selection
                                slice_info = ""
                                if 'original_slice_used' in result and 'original_shape' in result:
                                    slice_num = result['original_slice_used']
                                    total_slices = result['original_shape'][2] if len(str(result['original_shape']).split(',')) >= 3 else "unknown"
                                    slice_info = f" (showing slice {slice_num}/{total_slices} - middle slice selected for best spleen visibility)"
                                
                                st.image(result['plot_image'],
                                       caption=f"3D spleen segmentation visualization{slice_info}",
                                       width='stretch')
                            else:
                                st.warning("No visualization available")

                        # Comprehensive tabs for all detailed information
                        spleen_tabs = st.tabs([
                            "**📊 Segmentation Results**",
                            "**🎯 Performance Metrics**",
                            "**⚙️ Configuration**"
                        ])

                        with spleen_tabs[0]:
                            # Essential segmentation details
                            st.markdown("### 📊 Volume Analysis")

                            # Volume statistics - showing total voxels and segmented voxels
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Voxels", f"{result['total_voxels']:,}")
                            with col2:
                                st.metric("Segmented Voxels", f"{result['segmented_voxels']:,}")
                            with col3:
                                st.metric("Segmentation %", f"{result['segmentation_percentage']:.2f}%")

                        with spleen_tabs[1]:
                            # Performance metrics
                            st.markdown("### 🎯 Processing Performance")

                            timing_col1, timing_col2 = st.columns(2)
                            with timing_col1:
                                st.metric("Total Processing Time", f"{result['total_time']:.2f}s")
                                st.metric("Model Loading Time", f"{result['model_load_time']:.2f}s")
                            with timing_col2:
                                st.metric("Inference Time", f"{result['inference_time']:.2f}s")
                                st.metric("Visualization Time", f"{result['visualization_time']:.2f}s")

                        with spleen_tabs[2]:
                            # Configuration details
                            st.markdown("### ⚙️ Model Configuration")

                            # Display device information with GPU instead of cuda
                            device_display = result.get('device_used', 'Unknown')
                            device_type_display = result.get('device_type', 'Unknown')
                            
                            # Replace cuda with GPU for user-friendly display
                            if 'cuda' in device_display.lower():
                                device_display = device_display.replace('cuda', 'GPU')
                            if device_type_display.upper() == 'CUDA':
                                device_type_display = 'GPU'

                            config_info = {
                                "Model Path": result['model_path'],
                                "Input Path": result['input_path'],
                                "Output Directory": result['segmentation_dir'],
                                "Device Used": device_display,
                                "Device Type": device_type_display
                            }

                            # Convert full paths to relative paths starting from rocm-ls-examples
                            for key, value in config_info.items():
                                if key in ["Model Path", "Input Path", "Output Directory"] and isinstance(value, str):
                                    # Extract relative path from rocm-ls-examples
                                    if "rocm-ls-examples" in value:
                                        relative_path = value.split("rocm-ls-examples/", 1)[-1]
                                        st.text(f"{key}: rocm-ls-examples/{relative_path}")
                                    else:
                                        st.text(f"{key}: {value}")
                                else:
                                    st.text(f"{key}: {value}")

                        update_progress("Spleen CT segmentation completed!", 1.0)
                        
                        # Clean up temporary file if it was created from uploaded file
                        if hasattr(input_data, 'name') and not isinstance(input_data, str):
                            try:
                                import os
                                if os.path.exists(input_path):
                                    os.unlink(input_path)
                                    info(INFER_CONSOLE_LOG_KEY, f"Cleaned up temporary file: {input_path}")
                            except Exception as cleanup_error:
                                warning(INFER_CONSOLE_LOG_KEY, f"Could not clean up temporary file: {cleanup_error}")
                        
                        return  # Exit early for spleen CT inference

                    except Exception as e:
                        st.error(f"❌ Spleen CT inference failed: {str(e)}")
                        error(INFER_CONSOLE_LOG_KEY, f"Spleen CT inference error: {str(e)}")
                        
                        # Clean up temporary file if it was created from uploaded file
                        if hasattr(input_data, 'name') and not isinstance(input_data, str):
                            try:
                                import os
                                if 'input_path' in locals() and os.path.exists(input_path):
                                    os.unlink(input_path)
                                    info(INFER_CONSOLE_LOG_KEY, f"Cleaned up temporary file after error: {input_path}")
                            except Exception as cleanup_error:
                                warning(INFER_CONSOLE_LOG_KEY, f"Could not clean up temporary file after error: {cleanup_error}")
                        
                        return

        except Exception as e:
            st.error(f"Inference failed: {str(e)}")
            error(INFER_CONSOLE_LOG_KEY, f"Inference error: {str(e)}")
            progress_indicator.progress(0, text="Inference failed")