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
from components.io import render_markdown
from components.tooltips import tooltips
from components.console_log import console_log_init, console_log_view

from products.monai.training.sidebar import monai_training_sidebar
from products.monai.training.driver import monai_training_driver
from products.monai.inference.sidebar import monai_inference_sidebar
from products.monai.console import (
        TRAIN_CONSOLE_LOG_KEY, 
        INFER_CONSOLE_LOG_KEY
)
from products.monai.styles import (
        POWERED_BY_HTML, 
        POWERED_BY_CSS,
        MONAI_LAYOUT_CSS
)

# Common footer for both training and inference tabs
def monai_footer(console_log_key):
    st.markdown("---")
    st.markdown("###### 📋 Console Output")
    console_log_view(console_log_key)
    
    st.markdown("---")
    st.markdown("###### ⚡ System Info")
    cpu_details = get_cpu_info()
    gpu_details = get_gpu_info()
    cupy_details = get_package_version("amd-cupy")
    hipcim_details = get_package_version("amd-hipcim")
    monai_details = get_package_version("amd-monai")
    st.markdown(POWERED_BY_CSS, unsafe_allow_html=True)
    st.markdown(
        POWERED_BY_HTML.format(
            gpu=gpu_details,
            hipcim=hipcim_details,
            monai=monai_details,
            cupy=cupy_details,
        ),
        unsafe_allow_html=True
    )

def monai_layout():
    # Initialize the console log
    console_log_init(TRAIN_CONSOLE_LOG_KEY)
    
    # Apply custom styling
    st.markdown(MONAI_LAYOUT_CSS, unsafe_allow_html=True)
    
    # Create separate tabs for training and inference
    monai_training, monai_inference = st.tabs(["**🎯 Training**", "**🚀 Inference**"])

    with monai_training:
        training_sidebar, training_main, training_rsb = st.columns([1.2, 5.5, 1.8], gap="medium")

        # --- Sidebar Controls for Training ---
        with training_sidebar:
            monai_training_sidebar()

        # --- Main Training Panel ---
        with training_main:
            monai_training_driver()

        # --- Right Sidebar with Footer Info ---
        with training_rsb:
            render_markdown("markdown/monai_training_intro.md")
            monai_footer(TRAIN_CONSOLE_LOG_KEY)

    with monai_inference:
        inference_sidebar, inference_main, inference_rsb = st.columns([1.2, 5.5, 1.8], gap="medium")

        # --- Sidebar Controls for Inference ---
        with inference_sidebar:
            monai_inference_sidebar()

        # --- Main Inference Panel ---
        with inference_main:
            from products.monai.inference.driver import monai_inference_driver
            monai_inference_driver()

        # --- Right Sidebar with Footer Info ---
        with inference_rsb:
            render_markdown("markdown/monai_inference_intro.md")
            monai_footer(INFER_CONSOLE_LOG_KEY)

