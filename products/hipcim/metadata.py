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
import pandas as pd

from components.io import render_markdown
from components.utility import filesize_h, dl_datatype_to_str, unit_map
from components.diagnostics import exception, error, warning, note, info
from components.state import (
    session_state_get,
    session_state_set,
)
from components.styles import METADATA_TAB_CSS, METADATA_TAB_HTML

from products.hipcim.console import CONSOLE_LOG_KEY

from cucim import CuImage

# Display CuImage metadata
def display_metadata(cuImage):
    # Extract WSI metadata into session state
    # YXC order: spacing[1] is X, spacing[0] is Y
    session_state_set('spacing_x', cuImage.metadata['cucim']['spacing'][1]) 
    session_state_set('spacing_y', cuImage.metadata['cucim']['spacing'][0]) 
    session_state_set('unit_x', unit_map.get(cuImage.metadata['cucim']['spacing_units'][1]))
    session_state_set('unit_y', unit_map.get(cuImage.metadata['cucim']['spacing_units'][0]))

    # Get details of the image
    size_h = filesize_h(cuImage.path)
    width, height = cuImage.shape[1], cuImage.shape[0]
    pixel_type = dl_datatype_to_str(cuImage.dtype)
    width_physical = width * session_state_get('spacing_x')
    height_physical = height * session_state_get('spacing_y')

    st.markdown(METADATA_TAB_CSS, unsafe_allow_html=True)
    st.markdown(
        METADATA_TAB_HTML.format(
            filename = f"{cuImage.path}",
            img_size = f"{size_h}",
            img_width = f"{width}px ({width_physical:.2f} {st.session_state['unit_x']})",
            img_height = f"{height}px ({height_physical:.2f} {st.session_state['unit_y']})",
            device = f"GPU",
            px_type = f"{pixel_type}",
            dims = f"{cuImage.ndim} {cuImage.channel_names}"
        ),
        unsafe_allow_html=True
    )

def hipcim_metadata():
    # Tabify metadata for the original WSI, GPU tile and CPU tile
    wsi_intro, wsi_meta = st.tabs([
                                   "**Whole Slide Image (WSI)**",
                                   "**WSI Metadata**",
                                 ])
    with wsi_intro:
        # Introduce the WSI
        render_markdown("markdown/wsi_intro.md")

    with wsi_meta:
        # Display metadata of the WSI
        try:
            cuImage = session_state_get('cuImage')
            display_metadata(cuImage)
        except Exception as e:
            exception(CONSOLE_LOG_KEY, 
                      f"failed to load {session_state_get('selected_wsi')} using hipCIM: {e}")

