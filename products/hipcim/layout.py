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

from components.utility import (
    get_cpu_info,
    get_gpu_info,
    get_package_version
)
from components.tooltips import tooltips
from components.io import render_markdown
from components.styles import POWERED_BY_HTML, POWERED_BY_CSS
from components.console_log import console_log_init, console_log_view

from products.hipcim.sidebar import hipcim_sidebar
from products.hipcim.main import hipcim_main

CONSOLE_LOG_KEY="hipCIM_CONSOLE_LOG"

# Custom CSS for improved UI
HIPCIM_LAYOUT_CSS = """
<style>
/* Card container for tiles */
.tile-card {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px;
    margin: 8px 0;
    border: 1px solid #e8eaed;
    transition: box-shadow 0.2s ease;
}
.tile-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

/* Section headers */
.section-header {
    font-size: 14px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #00A3E0;
}

/* Info panel styling */
.info-panel {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 16px;
    border-left: 4px solid #00A3E0;
}

/* Console area styling */
.console-area {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 12px;
    margin-top: 16px;
}

/* Sidebar improvements */
[data-testid="stSidebar"] .stExpander {
    background: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 8px;
}

/* Tile container improvements */
.stImage {
    border-radius: 8px;
    overflow: hidden;
}

/* Better tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}
</style>
"""

def hipcim_layout():
    # Initialize the console log
    console_log_init(CONSOLE_LOG_KEY)
    
    # Apply custom CSS
    st.markdown(HIPCIM_LAYOUT_CSS, unsafe_allow_html=True)
    
    # Design the hipCIM top-level layout with better proportions
    hipcim_lsb_col, hipcim_main_col, hipcim_rsb_col = st.columns([1.2, 5.5, 1.8], gap="medium")

    # --- Sidebar Controls ---
    with hipcim_lsb_col:
        selected_wsi_filename, selected_wsi_filepath, tile_size, x, y = hipcim_sidebar()

    # --- Main Panel ---
    with hipcim_main_col:
        hipcim_main()

    # --- Right Sidebar with Footer Info ---
    with hipcim_rsb_col:
        # Info panel with styling
        with st.container():
            render_markdown("markdown/hipcim_intro.md")
        
        st.divider()
        
        # Console section
        st.markdown("###### 📋 Console Output")
        console_log_view(CONSOLE_LOG_KEY)
        
        st.divider()
        
        # System info footer
        st.markdown("###### ⚡ System Info")
       #cpu_details = get_cpu_info()
        gpu_details = get_gpu_info()
        hipcim_details = get_package_version("amd-hipcim")
        st.markdown(POWERED_BY_CSS, unsafe_allow_html=True)
        st.markdown(
            POWERED_BY_HTML.format(
               #cpu=cpu_details,
                gpu=gpu_details,
                hipcim=hipcim_details
            ),
            unsafe_allow_html=True
        )

