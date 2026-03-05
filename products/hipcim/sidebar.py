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
from streamlit_option_menu import option_menu  # for icon support

from components.diagnostics import error, exception, warning, note, info
from components.io import file_upload_widget, render_markdown
from components.tooltips import tooltips
from components.utility import get_image_shape
from components.state import (
    session_state_get,
    session_state_set,
)

from products.hipcim.console import CONSOLE_LOG_KEY
from products.hipcim.styles import COMPACT_SIDEBAR_CSS

from cucim import CuImage

# Professional transformation catalog with business context and technical explanations
OP_CATALOG = {
    "stain_separation": {
        "icon": "🧪",
        "display_name": "Stain Separation",
        "category": "Color Analysis",
        "business_use": "Isolate specific tissue components for cancer detection, cell counting, and biomarker quantification in pathology workflows.",
        "technical_desc": "Decomposes H&E or IHC-stained images into individual stain channels using color deconvolution in the HED color space.",
        "params": {"stain": ["hematoxylin", "eosin", "dab"]},
        "param_help": {
            "stain": "**Hematoxylin**: Nuclear stain (blue/purple) - identifies cell nuclei for tumor grading. **Eosin**: Cytoplasmic stain (pink) - reveals tissue architecture. **DAB**: Immunohistochemistry marker (brown) - detects specific proteins/antigens."
        }
    },
    "gabor_filter": {
        "icon": "🔬",
        "display_name": "Texture Analysis",
        "category": "Feature Extraction",
        "business_use": "Identify tissue patterns, fibrosis levels, and structural abnormalities for automated tissue classification.",
        "technical_desc": "Applies Gabor wavelets to detect oriented textures and patterns at specific frequencies and angles.",
        "params": {"frequency": (0.1, 0.9), "theta": (0, 180)},
        "param_help": {
            "frequency": "Controls pattern scale: lower values detect coarse textures (collagen bundles), higher values detect fine textures (cellular patterns).",
            "theta": "Orientation angle (degrees): 0° detects horizontal features, 90° detects vertical features."
        }
    },
    "sobel_edges": {
        "icon": "📐",
        "display_name": "Edge Detection",
        "category": "Feature Extraction",
        "business_use": "Delineate cell boundaries, tissue margins, and anatomical structures for segmentation tasks.",
        "technical_desc": "Computes intensity gradients using the Sobel operator to highlight boundaries between regions.",
        "params": {},
        "param_help": {}
    },
    "binary_dilation": {
        "icon": "⭕",
        "display_name": "Morphological Dilation",
        "category": "Morphology",
        "business_use": "Connect fragmented tissue regions, fill gaps in cell detection, and enhance region-of-interest masks.",
        "technical_desc": "Expands foreground regions using a disk-shaped structuring element after Otsu thresholding.",
        "params": {"iterations": (1, 10)},
        "param_help": {
            "iterations": "Dilation radius in pixels. Higher values create larger expansions, useful for connecting distant structures."
        }
    },
    "remove_small_objects": {
        "icon": "🧹",
        "display_name": "Noise Removal",
        "category": "Morphology",
        "business_use": "Eliminate imaging artifacts, debris, and false positives to improve downstream analysis accuracy.",
        "technical_desc": "Labels connected components and removes objects smaller than a specified pixel threshold.",
        "params": {"min_size": (10, 1000)},
        "param_help": {
            "min_size": "Minimum object size to retain (in pixels). Objects smaller than this threshold are removed as noise."
        }
    },
    "rotate": {
        "icon": "🔄",
        "display_name": "Image Rotation",
        "category": "Geometric Transform",
        "business_use": "Correct specimen orientation, augment training data, and standardize image alignment.",
        "technical_desc": "Rotates the image around its center using bilinear interpolation with zero-padding.",
        "params": {"angle": (0, 360)},
        "param_help": {
            "angle": "Rotation angle in degrees (counter-clockwise). Common values: 90°, 180°, 270° for orthogonal rotations."
        }
    },
    "warp_affine": {
        "icon": "🎯",
        "display_name": "Affine Transform",
        "category": "Geometric Transform",
        "business_use": "Perform geometric corrections, register multi-modal images, and generate augmented training samples.",
        "technical_desc": "Applies combined scale, rotation, shear, and translation transformations using affine matrix operations.",
        "params": {},
        "param_help": {}
    },
}

# Initialize hipCIM transformation pipeline
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = []

# Layout the hipCIM sidebar control panel
def hipcim_sidebar():
    render_markdown("markdown/hipcim_sidebar_header.md")
    selected_wsi_filename, selected_wsi_filepath = file_upload_widget()

    # Current tile size and positioning
    tile_max_dimension = session_state_get('tile_max_dimension')
    tile_size = session_state_get('tile_size')
    x = session_state_get('position_x')
    y = session_state_get('position_y')

    # Sliders in a compact expander
    st.markdown(COMPACT_SIDEBAR_CSS, unsafe_allow_html=True)
    with st.expander("**Tile Controls**", expanded=True):
        if not selected_wsi_filepath:
            error(CONSOLE_LOG_KEY,
                  "failed to retrieve wsi filepath (missing sample_images/?)")
        else:
            img_width, img_height = get_image_shape(selected_wsi_filepath)
            tile_size_initial = int(min(128, img_width // 2, img_height // 2))
            tile_size = st.slider(
                "Tile Size", 
                int(min(128, img_width/2, img_height/2)), 
                min(tile_max_dimension, img_width - 1, img_height - 1),
                tile_size_initial,
                help=tooltips['tile_size']
            )

            # Dynamically set initial X/Y centered on image 
            # based on uploaded image size and tile size
            x_initial = max(img_width // 2 - tile_size, 0)
            x_max = max(img_width - tile_size, 0)
            x = st.slider(
                "X position", 
                0, x_max, x_initial, step=32,
                help=tooltips['x_position']
            )
            y_initial = max(img_height // 2 - tile_size, 0)
            y_max = max(img_height - tile_size, 0)
            y = st.slider(
                "Y position", 
                0, y_max, y_initial, step=32,
                help=tooltips['y_position']
            )

            # Update session state with loaded image details
            session_state_set('selected_wsi', selected_wsi_filepath)
            session_state_set('wsi_width', img_width)
            session_state_set('wsi_height', img_height)
            session_state_set('tile_size', tile_size)
            session_state_set('position_x', x)
            session_state_set('position_y', y)

            # Squirrel off the CuImage into the session state
            try:
                cuImage = CuImage(session_state_get('selected_wsi'))
                session_state_set('cuImage', cuImage)
            except Exception as e:
                session_state_set('cuImage', None)
                exception(CONSOLE_LOG_KEY,
                          f"failed to load {session_state_get('selected_wsi')} using hipCIM: {e}")

    st.caption(f"📐 Tile: {tile_size}px  |  📍 Position: ({x}, {y})")

    # Pipeline builder UI with professional styling
    with st.expander("**🔬 Transformation Pipeline**", expanded=True):
        # Group operations by category for better organization
        categories = {}
        for op_key, op_info in OP_CATALOG.items():
            cat = op_info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((op_key, op_info))
        
        with st.expander("➕ Add Transformation Step", expanded=False):
            # Create operation selector with display names
            op_keys = list(OP_CATALOG.keys())
            op_display_names = [
                f"{OP_CATALOG[k]['icon']} {OP_CATALOG[k]['display_name']}" 
                for k in op_keys
            ]
            selected_idx = st.selectbox(
                "Select Operation", 
                list(range(len(op_keys))), 
                format_func=lambda i: op_display_names[i]
            )
            op_key = op_keys[selected_idx]
            op_info = OP_CATALOG[op_key]
            
            # Display business context and technical description
            st.markdown(f"**Category:** {op_info['category']}")
            st.info(f"💼 **Business Use:** {op_info['business_use']}")
            st.caption(f"🔧 *Technical:* {op_info['technical_desc']}")
            
            # Parameter configuration
            param_defs = op_info["params"]
            param_help = op_info["param_help"]
            params = {}

            if param_defs:
                st.markdown("**Parameters:**")
                for pname, pdef in param_defs.items():
                    help_text = param_help.get(pname, "")
                    if isinstance(pdef, tuple):  # range slider
                        params[pname] = st.slider(
                            pname.replace('_', ' ').title(), 
                            float(pdef[0]), float(pdef[1]),
                            float(pdef[0]), 
                            key=f"param_{op_key}_{pname}",
                            help=help_text
                        )
                    elif isinstance(pdef, list):
                        params[pname] = st.selectbox(
                            pname.replace('_', ' ').title(), 
                            pdef, 
                            key=f"param_{op_key}_{pname}",
                            help=help_text
                        )
                        if help_text:
                            st.caption(help_text)
            
            if st.button(
                f"➕ Add '{op_info['display_name']}' to Pipeline",
                type="primary",
                use_container_width=True
            ):
                step = {"op": op_key, "params": params.copy()}
                st.session_state['pipeline'].append(step)
                st.rerun()

        # Render active pipeline with enhanced display
        st.markdown("---")
        st.markdown("###### 📋 Active Pipeline")
        
        if not st.session_state['pipeline']:
            st.info("No transformations configured. Add steps above to build your image processing workflow.")
        else:
            for i, step in enumerate(st.session_state['pipeline']):
                op_key = step['op']
                op_info = OP_CATALOG[op_key]
                
                with st.expander(
                    f"**{i+1}.** {op_info['icon']} {op_info['display_name']}", 
                    expanded=False
                ):
                    # Show category badge
                    st.caption(f"📁 {op_info['category']}")
                    
                    # Show parameters if any
                    if step['params']:
                        st.markdown("**Configuration:**")
                        for pname, pval in step['params'].items():
                            st.text(f"  • {pname.replace('_', ' ').title()}: {pval}")
                    else:
                        st.text("  No parameters (default settings)")
                    
                    # Action buttons in a row
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        if st.button("🗑️ Remove", key=f"remove_{i}", use_container_width=True):
                            st.session_state['pipeline'].pop(i)
                            st.rerun()
                    with cols[1]:
                        if i > 0:
                            if st.button("⬆️ Up", key=f"up_{i}", use_container_width=True):
                                st.session_state['pipeline'][i-1], st.session_state['pipeline'][i] = (
                                    st.session_state['pipeline'][i], st.session_state['pipeline'][i-1]
                                )
                                st.rerun()
                    with cols[2]:
                        if i < len(st.session_state['pipeline']) - 1:
                            if st.button("⬇️ Down", key=f"down_{i}", use_container_width=True):
                                st.session_state['pipeline'][i+1], st.session_state['pipeline'][i] = (
                                    st.session_state['pipeline'][i], st.session_state['pipeline'][i+1]
                                )
                                st.rerun()
            
            # Pipeline summary
            st.markdown("---")
            st.caption(f"✅ {len(st.session_state['pipeline'])} transformation(s) configured")

            

    return selected_wsi_filename, selected_wsi_filepath, tile_size, x, y
