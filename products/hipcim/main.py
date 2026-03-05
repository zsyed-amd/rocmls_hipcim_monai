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
from PIL import Image
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import io
import os
from components.io import render_markdown
from components.styles import (
    THUMB_PLACEHOLDER_CSS,
    THUMB_PLACEHOLDER_HTML,
    TILE_PLACEHOLDER_CSS,
    TILE_PLACEHOLDER_HTML,
)
from components.utility import (
    generate_wsi_thumbnail,
    rescale_image,
)
from components.state import session_state_get

from products.hipcim.metadata import hipcim_metadata
from products.hipcim.tile import display_tiles

from products.hipcim.llm_pathology import LLaVANextPathologyAnalyzer, create_llava_analysis_ui, LLAVA_CONSOLE_LOG_KEY
from components.state import session_state_get, session_state_set


# Disable PIL.Image.DecompressionBombError: decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

def display_thumbnail_with_grid(thumb_path):
    # Load downsampled JPEG thumbnail
    thumb_np = np.array(Image.open(thumb_path).convert("RGB"))
    thumb_height, thumb_width = thumb_np.shape[:2]

    # Compute applicable scaling
    scale_x = thumb_width / session_state_get('wsi_width')
    scale_y = thumb_height / session_state_get('wsi_height')

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 2.9), dpi=96)
    # This yields image of 800x278 px (width x height)

    # Display thumbnail with ticks
    ax.imshow(thumb_np)
    ax.set_xticks(np.arange(0, thumb_width, step=100))
    ax.set_yticks(np.arange(0, thumb_height, step=100))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Add labels for the ticks
    for i in ax.get_xticks():
        ax.text(i, -15, str(int(i / scale_x)), ha='center', va='center', fontsize=6)
    for j in ax.get_yticks():
        ax.text(-15, j, str(int(j / scale_y)), ha='right', va='center', fontsize=6)

    # Highlight RoI rectangle
    x = session_state_get('position_x')
    y = session_state_get('position_y')
    tile_size = session_state_get('tile_size')
    rect = plt.Rectangle((x * scale_x, y * scale_y),
                         tile_size * scale_x, tile_size * scale_y,
                         linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.text(-120, thumb_height / 2, "Thumbnail Map", va='center', ha='right',
            rotation='vertical', fontsize=12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('on')

    # Save figure to PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Rescale the image using PIL to fit the placeholder without distortion
    image = Image.open(buf)
    image = rescale_image(image, max_width=800, max_height=250)

    # Render the thumbnail
    st.image(image) 

# Custom CSS for main panel tiles
MAIN_PANEL_CSS = """
<style>
.tile-container {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 12px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.tile-label {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
    margin-top: 8px;
    padding: 4px 8px;
    background: #f1f5f9;
    border-radius: 4px;
    display: inline-block;
}
.thumbnail-container {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    padding: 16px;
    border: 1px solid #e2e8f0;
}
.metadata-container {
    background: linear-gradient(135deg, #fafbfc 0%, #f0f4f8 100%);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e2e8f0;
    height: 100%;
}
</style>
"""

def hipcim_main():
    # Apply main panel CSS
    st.markdown(MAIN_PANEL_CSS, unsafe_allow_html=True)
    
    # hipCIM_r1: Thumbnail map with metadata panel
    st.markdown("##### 🔬 Whole Slide Image Overview")
    r1c1, r1c2 = st.columns([2.5, 1], gap="medium")

    with r1c1:
        current_wsi = session_state_get('selected_wsi')
        if current_wsi:
            # Generate the thumbnail from the WSI
            thumb_path = f".generated_thumbnails/{current_wsi}_thumb.jpg"
            with st.spinner('🔄 Generating thumbnail...'):
                generate_wsi_thumbnail(current_wsi, thumb_path, width=800)
            display_thumbnail_with_grid(thumb_path)
        else:
            st.markdown(THUMB_PLACEHOLDER_CSS, unsafe_allow_html=True)
            st.markdown(THUMB_PLACEHOLDER_HTML, unsafe_allow_html=True)

    with r1c2:
        hipcim_metadata()

    st.markdown("---")
    
    # hipCIM_r2: Display tiles side-by-side with section header
    st.markdown("##### 🖼️ Tile Comparison: Reference vs Transformed")
    display_tiles()
        # Get current processed image from session state



    cpu_tile = session_state_get('cpu_tile')
    gpu_tile = session_state_get('gpu_tile')
    transformation_pipeline = session_state_get('pipeline', [])
    
    if cpu_tile is not None or gpu_tile is not None:
        
        # Create tabs for different views
        image_tab, analysis_tab = st.tabs(["🖼️ **Image Processing**", "🔬 **LLaVA-NeXT Analysis**"])
        
        with image_tab:
            if cpu_tile is not None or gpu_tile is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**CPU Processed Image**")
                    # Debug: check what's in session state
                    cpu_transformed = session_state_get('transformed_cpu_tile_np')
                    if cpu_transformed is not None:
                        st.image(cpu_transformed, width=400)
                    else:
                        # Fallback: try to process the raw cpu_tile if available
                        if cpu_tile is not None:
                            try:
                                # Apply the same processing as in tile.py
                                from products.hipcim.transforms import apply_pipeline
                                transformed_cpu_tile = apply_pipeline(cpu_tile)
                                transformed_cpu_tile_np = np.asarray(transformed_cpu_tile)
                                
                                # Handle float images
                                if np.issubdtype(transformed_cpu_tile_np.dtype, np.floating):
                                    transformed_cpu_tile_np = np.clip(transformed_cpu_tile_np, 0, 1)
                                    transformed_cpu_tile_np = (transformed_cpu_tile_np * 255).astype(np.uint8)
                                elif transformed_cpu_tile_np.dtype != np.uint8:
                                    transformed_cpu_tile_np = transformed_cpu_tile_np.astype(np.uint8)
                                
                                st.image(transformed_cpu_tile_np, width=400)
                            except Exception as e:
                                st.error(f"Error processing CPU image: {e}")
                        else:
                            st.info("No CPU processed image available")
                
                with col2:
                    st.markdown("**GPU Processed Image**")
                    # Debug: check what's in session state
                    gpu_transformed = session_state_get('transformed_gpu_tile_np')
                    if gpu_transformed is not None:
                        st.image(gpu_transformed, width=400)
                    else:
                        # Fallback: try to process the raw gpu_tile if available
                        if gpu_tile is not None:
                            try:
                                # Apply the same processing as in tile.py
                                from products.hipcim.transforms import apply_pipeline
                                transformed_gpu_tile = apply_pipeline(gpu_tile, "cuda")
                                transformed_gpu_tile_np = cp.asnumpy(transformed_gpu_tile)
                                
                                # Handle float images
                                if np.issubdtype(transformed_gpu_tile_np.dtype, np.floating):
                                    transformed_gpu_tile_np = np.clip(transformed_gpu_tile_np, 0, 1)
                                    transformed_gpu_tile_np = (transformed_gpu_tile_np * 255).astype(np.uint8)
                                elif transformed_gpu_tile_np.dtype != np.uint8:
                                    transformed_gpu_tile_np = transformed_gpu_tile_np.astype(np.uint8)
                                
                                st.image(transformed_gpu_tile_np, width=400)
                            except Exception as e:
                                st.error(f"Error processing GPU image: {e}")
                        else:
                            st.info("No GPU processed image available")
                
                if transformation_pipeline:
                    st.markdown("**Applied Transformations:**")
                    for i, transform in enumerate(transformation_pipeline, 1):
                        # Handle the transform structure from transforms.py
                        if isinstance(transform, dict):
                            # Look for the correct keys used in transforms.py
                            op_name = transform.get('op', transform.get('name', transform.get('operation', 'Unknown')))
                            params = transform.get('params', transform.get('parameters', {}))
                            
                            # Map operation names to display names
                            op_display_names = {
                                "stain_separation": "Stain Separation",
                                "gabor_filter": "Gabor Filter",
                                "sobel_edges": "Sobel Edge Detection", 
                                "binary_dilation": "Binary Dilation",
                                "remove_small_objects": "Remove Small Objects",
                                "rotate": "Rotation",
                                "warp_affine": "Affine Warp"
                            }
                            
                            display_name = op_display_names.get(op_name, op_name)
                            
                            if params:
                                param_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                                st.markdown(f"{i}. **{display_name}** ({param_str})")
                            else:
                                st.markdown(f"{i}. **{display_name}**")
                        elif isinstance(transform, str):
                            st.markdown(f"{i}. **{transform}**")
                        else:
                            transform_name = getattr(transform, 'name', str(transform))
                            st.markdown(f"{i}. **{transform_name}**")
                else:
                    st.info("No transformations applied yet. Add transformations using the pipeline controls above.")
            else:
                st.info("Process an image using the controls above to see results here.")
            
        with analysis_tab:
            st.markdown("#### Analyze Transformed WSI with LLaVA-NeXT")
            
            # LLaVA-NeXT Configuration UI
            model_name, custom_prompt, temperature, max_tokens, do_sample = create_llava_analysis_ui()
            #model_name = "llava-hf/llava-1.5-7b-hf"
            
            # Analysis controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                image_source = st.radio(
                    "Select Image for Analysis",
                    ["GPU Processed", "CPU Processed"],
                    horizontal=True
                )
                
            with col2:
                # Check if model is loaded before enabling analyze button
                model_loaded = session_state_get('model_loaded', False)
                analyze_button = st.button(
                    "🔬 Analyze with LLaVA-NeXT",
                    type="primary",
                    disabled=not model_loaded,
                    use_container_width=True,
                    help="Load the model first to enable analysis" if not model_loaded else "Analyze the selected image"
                )
            
            # Memory management controls
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🧠 Load Model", help="Pre-load model into memory"):
                    with st.spinner("Loading LLaVA-NeXT model..."):
                        try:
                            # Clear any existing error state
                            if 'model_load_error' in st.session_state:
                                del st.session_state['model_load_error']
                            
                            # Clear any existing analyzer to force fresh load
                            session_state_set('llava_analyzer', None)
                            session_state_set('model_loaded', False)
                            
                            from products.hipcim.llm_pathology import LLaVANextPathologyAnalyzer
                            
                            st.info(f"Creating fresh analyzer for: {model_name}")
                            analyzer = LLaVANextPathologyAnalyzer(model_name)
                            
                            # Force a fresh load by ensuring loaded flag is False
                            analyzer.loaded = False
                            
                            st.info("Calling load_model()...")
                            
                            # Temporarily override the log methods to capture them in Streamlit
                            captured_logs = []
                            original_log_info = analyzer.log_info
                            original_log_error = analyzer.log_error
                            
                            def capture_log_info(msg):
                                captured_logs.append(f"INFO: {msg}")
                                st.info(msg)
                                return original_log_info(msg)
                                
                            def capture_log_error(msg):
                                captured_logs.append(f"ERROR: {msg}")
                                st.error(msg)
                                return original_log_error(msg)
                            
                            analyzer.log_info = capture_log_info
                            analyzer.log_error = capture_log_error
                            
                            success = analyzer.load_model()
                            
                            # Restore original methods
                            analyzer.log_info = original_log_info
                            analyzer.log_error = original_log_error
                            
                            st.info(f"load_model() returned: {success}")
                            st.info(f"Model object: {'Loaded' if analyzer.model else 'None'}")
                            st.info(f"Processor object: {'Loaded' if analyzer.processor else 'None'}")
                            
                            if success and analyzer.model is not None and analyzer.processor is not None:
                                st.success("✅ Model loaded successfully!")
                                session_state_set('llava_analyzer', analyzer)
                                session_state_set('model_loaded', True)
                                st.rerun()
                            else:
                                st.error("❌ Failed to load model - model or processor is None")
                                session_state_set('model_loaded', False)
                                st.session_state['model_load_error'] = f"Model loading returned {success} but objects not created properly"
                                
                                # Show captured logs
                                with st.expander("Captured Logs"):
                                    for log in captured_logs:
                                        st.text(log)
                                
                                # Show detailed state for debugging
                                with st.expander("Debug Info"):
                                    st.write(f"Success flag: {success}")
                                    st.write(f"Analyzer.loaded: {getattr(analyzer, 'loaded', 'Unknown')}")
                                    st.write(f"Model is None: {analyzer.model is None}")
                                    st.write(f"Processor is None: {analyzer.processor is None}")
                                
                        except Exception as e:
                            error_msg = f"Error loading model: {str(e)}"
                            st.error(f"❌ {error_msg}")
                            session_state_set('model_loaded', False)
                            st.session_state['model_load_error'] = error_msg
                            
                            # Show detailed error in expander
                            with st.expander("Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
            
            with col4:
                if st.button("🗑️ Unload Model", help="Free GPU memory"):
                    try:
                        analyzer = session_state_get('llava_analyzer')
                        if analyzer:
                            analyzer.unload_model()
                            st.success("✅ Model unloaded, memory freed")
                        session_state_set('llava_analyzer', None)
                        session_state_set('model_loaded', False)
                        if 'model_load_error' in st.session_state:
                            del st.session_state['model_load_error']
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error unloading model: {e}")

            # Show current model status
            model_loaded = session_state_get('model_loaded', False)
            if model_loaded:
                st.success("🎯 Model is loaded and ready for analysis")
            elif 'model_load_error' in st.session_state:
                st.error(f"❌ {st.session_state['model_load_error']}")
            else:
                st.info("💡 Load the model first to enable analysis")
            
                        # Perform analysis when button clicked
            if analyze_button and model_loaded:
                
                # Select image based on user choice
                if image_source == "GPU Processed":
                    # Try to get transformed GPU image first, fallback to raw
                    selected_image = session_state_get('transformed_gpu_tile_np')
                    if selected_image is None:
                        selected_image = gpu_tile
                else:
                    # Try to get transformed CPU image first, fallback to raw
                    selected_image = session_state_get('transformed_cpu_tile_np')
                    if selected_image is None:
                        selected_image = cpu_tile
                
                if selected_image is not None:
                    
                    # Create transformation info string - fix the transform name extraction
                    if transformation_pipeline:
                        transform_names = []
                        for transform in transformation_pipeline:
                            if isinstance(transform, dict):
                                op_name = transform.get('op', 'Unknown')
                                # Map operation names to display names
                                op_display_names = {
                                    "stain_separation": "Stain Separation",
                                    "gabor_filter": "Gabor Filter",
                                    "sobel_edges": "Sobel Edge Detection", 
                                    "binary_dilation": "Binary Dilation",
                                    "remove_small_objects": "Remove Small Objects",
                                    "rotate": "Rotation",
                                    "warp_affine": "Affine Warp"
                                }
                                display_name = op_display_names.get(op_name, op_name)
                                transform_names.append(display_name)
                            else:
                                transform_names.append(str(transform))
                        transform_info = "Applied transformations: " + " → ".join(transform_names)
                    else:
                        transform_info = "No transformations applied"
                    
                    with st.spinner("🤖 Analyzing image..."):
                        
                        # Get analyzer from session state
                        analyzer = session_state_get('llava_analyzer')
                        
                        if analyzer:
                            try:
                                # Show analysis details
                                st.info(f"Analyzing {image_source.lower()} image...")
                                st.info(f"Image shape: {selected_image.shape}")
                                st.info(f"Model: {analyzer.model_name}")
                                
                                # Perform analysis with debugging
                                analysis_result = analyzer.analyze_pathology_image(
                                    image=selected_image,
                                    transformation_info=transform_info,
                                    custom_prompt=custom_prompt,
                                    temperature=temperature,
                                    max_new_tokens=max_tokens,
                                    do_sample=do_sample
                                )
                                
                                # Show the raw result for debugging
                                st.write(f"Raw analysis result: {analysis_result}")
                                st.write(f"Result type: {type(analysis_result)}")
                                st.write(f"Result length: {len(analysis_result) if analysis_result else 0}")
                                
                                if analysis_result and len(analysis_result.strip()) > 0:
                                    # Display results
                                    st.markdown("#### 📋 **Vision Model Analysis Report**")
                                    
                                    # Create expandable sections for better organization
                                    with st.expander("**Full AI Analysis**", expanded=True):
                                        st.markdown(analysis_result)
                                    
                                    # Analysis metadata
                                    with st.expander("**Analysis Details**", expanded=False):
                                        st.markdown(f"**Model Used:** {analyzer.model_name}")
                                        st.markdown(f"**Temperature:** {temperature}")
                                        st.markdown(f"**Max Tokens:** {max_tokens}")
                                        st.markdown(f"**Sampling:** {do_sample}")
                                        st.markdown(f"**Image Source:** {image_source}")
                                        st.markdown(f"**Transformations:** {transform_info}")
                                        
                                        # GPU memory info
                                        try:
                                            import torch
                                            if torch.cuda.is_available():
                                                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                                                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                                                st.markdown(f"**GPU Memory:** {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                                        except:
                                            pass
                                    
                                    # Save analysis to session state for reuse
                                    session_state_set('latest_llava_analysis', {
                                        'result': analysis_result,
                                        'image_source': image_source,
                                        'transformations': transform_info,
                                        'model': analyzer.model_name,
                                        'temperature': temperature,
                                        'timestamp': st.session_state.get('current_time', 'Unknown')
                                    })
                                    
                                else:
                                    st.error("❌ Analysis failed. The model returned no results.")
                                    
                                    # Show captured logs for debugging
                                    from components.console_log import console_log_view
                                    logs = console_log_view('llava_pathology_log')
                                    if logs:
                                        with st.expander("Debug Logs"):
                                            for log in logs[-10:]:  # Show last 10 logs
                                                st.text(log)
                                    
                                    # Test with a very simple prompt
                                    st.info("Trying with simplified analysis...")
                                    simple_result = analyzer.analyze_pathology_image(
                                        image=selected_image,
                                        custom_prompt="What is in this image?",
                                        max_new_tokens=50,
                                        temperature=0.7
                                    )
                                    
                                    if simple_result:
                                        st.success(f"Simple test result: {simple_result}")
                                    else:
                                        st.error("Even simple analysis failed")
                                    
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                with st.expander("Error Details"):
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            st.error("❌ No analyzer found in session state. Please reload the model.")
                
                else:
                    st.warning("⚠️ No processed image available for analysis.")
            
            # Show previous analysis if available
            previous_analysis = session_state_get('latest_llava_analysis')
            if previous_analysis and not analyze_button:
                st.markdown("#### 📚 **Previous LLaVA-NeXT Analysis**")
                with st.expander("**Last Analysis Result**", expanded=False):
                    st.markdown(f"**Model:** {previous_analysis.get('model', 'Unknown')}")
                    st.markdown(f"**Image Source:** {previous_analysis['image_source']}")
                    st.markdown(f"**Transformations:** {previous_analysis['transformations']}")
                    st.markdown("**Analysis:**")
                    st.markdown(previous_analysis['result'])