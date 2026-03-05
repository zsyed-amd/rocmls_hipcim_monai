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
import torch
import os
import time
import glob
import sys
from pathlib import Path

# Add project root to Python path for imports
current_file = Path(__file__).resolve()
project_root = None
for parent in current_file.parents:
    if parent.name == "rocm-ls-examples":
        project_root = parent
        break
if project_root:
    sys.path.insert(0, str(project_root))

from components.io import file_upload_widget, render_markdown
from components.diagnostics import error, warning, note, info
from components.state import (
    session_state_get,
    session_state_set,
)
from components.path_utils import resolve_path
from products.monai.training.models.zoo import MODEL_ZOO
from products.monai.console import INFER_CONSOLE_LOG_KEY
from products.monai.inference.tooltips import tooltips
from products.monai.inference.data_manager import (
    get_data_info,
    download_sample_data,
    download_model,
    get_available_data_files,
    get_available_model_folders,
    get_model_files_for_folder
)

# Layout the MONAI sidebar control panel
def monai_inference_sidebar():
    render_markdown("markdown/monai_inference_sidebar_header.md")

    # Device selection for inference
    device_selection_disabled = True
    device_index = 2
    if torch.cuda.is_available():
        device_selection_disabled = False
        device_index = 0
    else:
        warning(INFER_CONSOLE_LOG_KEY, "No GPU detected! Device selection disabled")

    selected_device = st.radio(
        label="**Select Device**",
        options=['Auto', 'GPU', 'CPU'],
        index=device_index,
        horizontal=True,
        disabled=device_selection_disabled,
        key="monai_inference_device",
        help=tooltips['device_select'],
    )

    # Setup inference device
    if selected_device == "Auto":
        selected_device = "GPU" if torch.cuda.is_available() else "CPU"
    if selected_device == "GPU":
        inference_device = torch.device("cuda")
    else:
        inference_device = torch.device("cpu")

    session_state_set("inference_device", inference_device)
    session_state_set("inference_device_type", selected_device)

    # Model source selection
    st.markdown("**Model Source**")
    model_source = st.radio(
        "Choose model source:",
        options=["Custom"],
        index=0,
        horizontal=True,
        key="monai_model_source",
        help=tooltips['model_source'],
    )

    selected_model = None
    model_info = None

    # Custom trained model selection - simplified approach
    st.markdown("**Step 1: Select Model Type**")
    
    # Get available model folders from demo/custom_trained
    model_folders = get_available_model_folders()
    
    if not model_folders:
        st.warning("❌ No model folders found in `demo/custom_trained/`")
        st.info("💡 Please ensure your trained models are organized in: `demo/custom_trained/{model_type}/models/`")
        return

    selected_folder_name = st.selectbox(
        "Choose model folder:",
        model_folders,
        key="monai_model_folder_select"
    )

    if selected_folder_name:
        # Step 2: Select specific model file
        st.markdown("**Step 2: Select Model Weight File**")
        
        model_files = get_model_files_for_folder(selected_folder_name)
        
        if not model_files:
            st.warning(f"❌ No model files found in `demo/custom_trained/{selected_folder_name}/models/`")
            return

        # Create display names for the model files
        model_display_options = []
        for model_path in model_files:
            filename = os.path.basename(model_path)
            model_display_options.append(f"{filename}")

        selected_model_display = st.selectbox(
            "Choose model file:",
            model_display_options,
            key="monai_model_file_select"
        )

        if selected_model_display:
            # Find the corresponding full path
            selected_model_path = None
            for model_path in model_files:
                if os.path.basename(model_path) == selected_model_display:
                    selected_model_path = model_path
                    break

            if selected_model_path:
                # Determine model type from folder name
                model_type = 'generic'
                folder_name_lower = selected_folder_name.lower()
                # Check pathology FIRST (before ct check) since pathology_tumor_detection contains "ct"
                if 'pathology' in folder_name_lower or 'tumor' in folder_name_lower:
                    model_type = 'pathology'
                elif 'spleen' in folder_name_lower or 'ct' in folder_name_lower:
                    model_type = 'spleen_ct'

                # Create model info for the custom model
                model_info = {
                    'name': f"Custom: {selected_folder_name}",
                    'source': 'custom',
                    'file_path': selected_model_path,  # Use file_path for consistency with driver
                    'model_type': model_type,
                    'id': selected_folder_name
                }

                st.success(f"✅ Selected: {selected_model_display}")

                # Set selected_model so it's available outside the nested condition
                selected_model = selected_folder_name


    # Inference configuration for pathology models
    if model_info and (model_info.get('source') == 'custom' and
                      ('pathology' in model_info.get('model_type', '').lower() or
                       'pathology' in model_info.get('file_path', '').lower())):

        # Pathology configuration with MONAI training styling
        st.markdown(
            "<h6 style='font-family: Arial, sans-serif; color: #333;'>Pathology Configuration</h6>",
            unsafe_allow_html=True
        )

        # Tile size configuration
        tile_size = st.slider(
            "**Tile size (pixels)**",
            min_value=224,
            max_value=20000,  # Increased max to 20000
            value=224,       # Keep 224 as default (standard for pathology)
            step=224,
            help="Size of each tile for analysis. 224 recommended for pathology models. Larger values need more memory.",
            key="pathology_tile_size"
        )

        # Batch size configuration
        batch_size = st.selectbox(
            "**Batch processing**",
            #options=[1, 2, 4, 8, 16, 32, 64, 128],
            #index=2,  # Default to 4
            #help="Number of tiles per batch. Higher values = faster processing.",
            options=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            index=9,  # Default to 512
            help="Number of tiles per batch. Higher values = faster GPU utilization. 512 recommended for most GPUs. Use 1024-2048 if you have 64GB+ GPU memory.",
            key="pathology_batch_size"
            
        )

        # Confidence threshold configuration
        confidence_threshold = st.slider(
            "**Confidence threshold**",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score to classify a patch as tumor. Default is 0.5.",
            key="pathology_confidence_threshold"
        )

    # Data input section - always show all three options
    st.markdown("**Input Data**")

    # Always show all three data source options
    data_source_options = ["Local Data", "Upload File", "Paste Path"]
    default_index = 0

    data_source = st.radio(
        "Data source:",
        options=data_source_options,
        index=default_index,
        horizontal=True,
        key="monai_data_source"
    )

    input_data = None
    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload medical image",
            type=['nii', 'nii.gz', 'dcm', 'png', 'jpg', 'tiff'],
            key="monai_upload_file"
        )
        if uploaded_file is not None:
            input_data = uploaded_file
    elif data_source == "Local Data":
        # Local data selection from model-specific folders
        available_files = []

        # Try to get files based on current model selection
        # ALWAYS use the most recent model_info first (from current sidebar run)
        current_model_info = model_info
        if not current_model_info:
            current_model_info = session_state_get('monai_inference_model_info')


        if current_model_info:
            if current_model_info.get('source') == 'zoo':
                model_id = current_model_info['id']
                available_files = get_available_data_files(model_id)
            elif current_model_info.get('source') == 'custom':
                # For custom models, map model type to compatible data sources
                model_type = current_model_info.get('model_type', 'generic')
                if model_type == 'spleen_ct':
                    available_files = get_available_data_files('spleen_ct_seg')
                elif model_type == 'pathology':
                    available_files = get_available_data_files('pathology_tumor_detection')
                else:
                    available_files = get_available_data_files(current_model_info.get('id', ''))
        else:
            # If no model selected yet, start with empty list to force model selection first
            available_files = []

        if available_files:
            # Filter and prioritize files based on model type
            if current_model_info:
                model_type = current_model_info.get('model_type', 'generic')
                filtered_files = []

                for file_path in available_files:
                    file_name = os.path.basename(file_path).lower()

                    # For pathology models, ONLY show .tif files
                    if model_type == 'pathology':
                        if file_name.endswith(('.tif', '.tiff')):
                            filtered_files.append(file_path)
                    # For spleen CT models, ONLY show .nii files
                    elif model_type == 'spleen_ct':
                        if file_name.endswith(('.nii', '.nii.gz')):
                            filtered_files.append(file_path)
                    # For other models, show compatible files
                    else:
                        filtered_files.append(file_path)

                available_files = filtered_files[:5]  # Limit to top 5 most relevant

                # Simple file list without complex grouping
                display_options = []
                file_map = {}

                for file_path in available_files:
                    file_name = os.path.basename(file_path)
                    # Show just filename with size - resolve path for size calculation
                    try:
                        absolute_path = resolve_path(file_path)
                        file_size = os.path.getsize(absolute_path) / (1024 * 1024)  # MB
                        display_name = f"{file_name} ({file_size:.1f}MB)"
                    except:
                        display_name = file_name

                    display_options.append(display_name)
                    file_map[display_name] = file_path
                
                # Use model type in key to reset selectbox when model changes
                model_key = current_model_info.get('model_type', 'none') if current_model_info else 'none'
                selected_display = st.selectbox(
                    "Select local data:",
                    display_options,
                    key=f"monai_local_data_{model_key}"
                )
                
                if selected_display:
                    input_data = file_map[selected_display]
        else:
            # No files available - could be no model selected or no data for selected model
            if not current_model_info:
                st.info("💡 Please select a model first to see compatible data files")
            else:
                st.markdown("❌ No compatible data files found. See <a href=\"https://github.com/ROCm-LS/examples/blob/main/README.md#dataset-preparation\">README</a> for dataset setup instructions.", unsafe_allow_html=True)
                # Show helpful information about expected locations
                with st.expander("📁 Expected Data Locations"):
                    st.markdown("""
                    **Primary demo data locations:**
                    - `demo/data/spleen_ct_segmentation/`
                    - `demo/data/pathology_tumor_detection/`

                    **Custom trained models data:**
                    - `demo/custom_trained/spleen_ct_segmentation/`
                    - `demo/custom_trained/pathology_tumor_detection/`

                    **Model Zoo models (downloaded):**
                    - `demo/model_zoo_models/spleen_ct_segmentation/`
                    - `demo/model_zoo_models/pathology_tumor_detection/`
                    """)

    elif data_source == "Paste Path":
        # Manual path input
        st.markdown("**Enter file path:**")
        pasted_path = st.text_input(
            "File path",
            placeholder="e.g., /path/to/your/data/file.tif or demo/data/pathology_tumor_detection/tumor_001.tif",
            key="monai_paste_path"
        )

        if pasted_path.strip():
            # Resolve the path (convert relative to absolute if needed)
            try:
                absolute_path = resolve_path(pasted_path.strip())
                if os.path.exists(absolute_path):
                    input_data = absolute_path
                    st.success(f"✅ Found: {os.path.basename(absolute_path)}")
                else:
                    st.error(f"❌ File not found: {absolute_path}")
            except Exception as e:
                st.error(f"❌ Invalid path: {str(e)}")

    # Store all selections in session state immediately when available
    if selected_model:
        session_state_set('monai_inference_model', selected_model)
    if model_info:
        session_state_set('monai_inference_model_info', model_info)
    if input_data:
        session_state_set('monai_inference_input_data', input_data)

    # Status indicators
    if model_info:
        st.success(f"✅ Model: {model_info['name']}")

    if input_data:
        if isinstance(input_data, str):
            filename = os.path.basename(input_data)
        else:
            filename = getattr(input_data, 'name', 'uploaded file')
        st.success(f"✅ Data: {filename}")

    return model_info is not None and input_data is not None
