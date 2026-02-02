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
# Style strings for various downstream components
#
# Usage example:
#   from components.styles import CONSOLE_LOG_CSS, IMAGE_PLACEHOLDER_CSS
#   import streamlit as st
#   
#   st.markdown(CONSOLE_LOG_CSS, unsafe_allow_html=True)
#   # ... your console log code
#   
#   st.markdown(IMAGE_PLACEHOLDER_CSS, unsafe_allow_html=True)
#   # ... your image placeholder code
#

# Console logging 
CONSOLE_LOG_CSS = """
<style>
.console-output {
    height: 140px;
    min-height: 90px;
    max-height: 220px;
    overflow-y: auto;
    background: #1a1a1a;
    color: #e6e6e6;
    font-family: 'Fira Mono', 'Consolas', Monospace;
    font-size: 10px;
    border-radius: 7px;
    border: 1px solid #333;
    padding: 4px 4px;
    white-space: pre-wrap;
}
.stButton button {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 14px;
  font-family: -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
  border-radius: 6px;
  border: none;

  color: #fff;
  background: linear-gradient(180deg, #4B91F7 0%, #367AF6 100%);
   background-origin: border-box;
  box-shadow: 0px 0.5px 1.5px rgba(54, 122, 246, 0.25), inset 0px 0.8px 0px -0.25px rgba(255, 255, 255, 0.2);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
}

.stDownloadButton button {
  width: 50px !important;
  height: 50px !important;
  border: none !important;
  border-radius: 50% !important;
  background-color: rgb(27, 27, 27) !important;
  color: #fff !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  position: relative !important;
  box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.11) !important;
  font-size: 24px !important;
  transition-duration: .3s !important;
  padding: 0 !important;
}

.stDownloadButton button:hover {
  background-color: rgb(150, 94, 255) !important;
  transition-duration: .3s !important;
}

.stDownloadButton button .download-icon {
  font-size: 32px;
  color: rgb(214, 178, 255);
  transition: color .3s;
}

.stDownloadButton button:hover .download-icon {
  color: #fff;
  animation: slide-in-top 0.6s cubic-bezier(0.250,0.460,0.450,0.940) both;
}

.stDownloadButton button .tooltip {
  position: absolute;
  top: 50%;
  left: 110%;
  opacity: 0;
  background-color: rgb(12, 12, 12);
  color: white;
  padding: 5px 10px;
  border-radius: 5px;
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition-duration: .3s;
  pointer-events: none;
  letter-spacing: 0.5px;
  z-index: 10;
  transform: translateY(-50%);
  white-space: nowrap;
}

.stDownloadButton button .tooltip::before {
  position: absolute;
  content: "";
  width: 10px;
  height: 10px;
  background-color: rgb(12, 12, 12);
  transform: rotate(45deg);
  left: -5px;
  top: 50%;
  margin-top: -5px;
}

.stDownloadButton button:hover .tooltip {
  opacity: 1;
  transition-duration: .3s;
}

@keyframes slide-in-top {
  0% {
    transform: translateY(-10px);
    opacity: 0;
  }
  100% {
    transform: translateY(0px);
    opacity: 1;
  }
}

.console-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-bottom: 0px;
}

/* Token colors for diagnostics */
.log-error      { color: #ff4c4c; font-weight: bold; }
.log-warning    { color: orange; font-weight: bold; }
.log-info       { color: #36aeea; font-weight: bold; }
.log-note       { color: #7acc5b; font-weight: bold; }
.log-debug      { color: #888; font-family: monospace; }
</style>
"""
CONSOLE_LOG_HTML = "<div class='console-output'>{content}</div>"

# hipCIM Thumbnail placeholder
THUMB_PLACEHOLDER_CSS = """
<style>
.thumb-placeholder {
    width: 100%;
    max-width: 100%;
    height: 250px;
    background: #e2e6ea;
    border-radius: 8px;
    border: 1px solid #dde2e8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2em;
    color: #666;
    overflow: hidden;
    position: relative;
}
</style>
"""
THUMB_PLACEHOLDER_HTML = """
<div class="thumb-placeholder">
Image Placeholder<br>(280px high, responsive)
</div>
"""

# hipCIM Tile placeholder
TILE_PLACEHOLDER_CSS = """
<style>
.processed-image-placeholder {
    background: #f6f9fc;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 160px;
    border-radius: 8px;
    border: 1px solid #e2e6ea;
    font-size: 1.1em;
    color: #bbb;
}
</style>
"""
TILE_PLACEHOLDER_HTML="<div class='processed-image-placeholder'>{label}</div>"

# hipCIM right sidebar footer
POWERED_BY_CSS = """
<style>
.powered-by-footer {
    color: #4b5563;
    font-size: 11px;
    line-height: 1.6;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #7dd3fc;
}
.powered-by-footer strong {
    color: #0284c7;
}
</style>
"""
POWERED_BY_HTML = """
<div class="powered-by-footer">
  <small>
    <strong> GPU:</strong> {gpu}<br>
    <strong>🧬 hipCIM:</strong> {hipcim}
  </small>
</div>
"""

# hipCIM metadata tab
METADATA_TAB_CSS = """
<style>
.metadata-tab {
    color: #374151;
    font-size: 12px;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    line-height: 1.8;
    background: #f8fafc;
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 3px solid #00A3E0;
}
.metadata-tab strong {
    color: #1f2937;
    font-weight: 600;
}
</style>
"""
METADATA_TAB_HTML = """
<div class="metadata-tab">
    <strong>📁 Filename:</strong> {filename}<br>
    <strong>💾 Size:</strong> {img_size}<br>
    <strong>↔️ Width:</strong> {img_width}<br>
    <strong>↕️ Height:</strong> {img_height}<br>
    <strong>🖥️ Device:</strong> {device}<br>
    <strong>🎨 Pixel type:</strong> {px_type}<br>
    <strong>📊 Dimensions:</strong> {dims}<br>
</div>
"""
