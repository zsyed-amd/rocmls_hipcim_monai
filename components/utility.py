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
import os
try:
    import pyvips
except Exception:
    pyvips = None
import re
import subprocess
import importlib.metadata
from collections import Counter
from PIL import Image
from components.diagnostics import exception, error, warning, info, note
from components.state import session_state_get

def open_and_convert_image(path):
    img = Image.open(path)
    if img.mode == 'I;16' or img.mode == 'I;16B':
        # Convert to 8-bit grayscale
        img = img.point(lambda i: i * (1. / 256)).convert('L')
    elif img.mode not in ('RGB', 'RGBA', 'L'):
        # Convert other modes to RGB safely
        img = img.convert('RGB')
    return img

def generate_wsi_thumbnail(input_path, output_path, width):
    """
    Generate a downscaled thumbnail of a WSI using pyvips, preserving aspect ratio.
    Writes to output_path if not already present.
    Creates output directory automatically if not present.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    try:
        if not os.path.exists(output_path):
            image = pyvips.Image.thumbnail(input_path, width)
            image.write_to_file(output_path)
    except Exception as e:
        # Optionally fallback to PIL
        try:
            warning(session_state_get('current_console_log_key'), 
                    "falling back on PIL to generate thumbnail")
            img = open_and_convert_image(input_path)
            img.thumbnail((width, width))
            img.save(output_path)
        except Exception as pil_e:
            exception(session_state_get('current_console_log_key'), 
                      f"thumbnail generation failed using pyvips/PIL: {e}; {pil_e}")

def rescale_image(image, max_width, max_height):
    """
    Resize PIL image so it fits within (max_width, max_height),
    preserving aspect ratio.
    Returns the resized PIL Image.
    """
    original_width, original_height = image.size
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_cpu_info():
    """
    Extracts CPU model, core/thread count, and frequency (Linux only).
    Returns a human-readable string.
    """
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        model = re.search(r'model name\s+: (.*)', cpuinfo).group(1)
        processors = re.findall(r'processor\s+: \d+', cpuinfo)
        physical_cores = len(processors)
        threads = len(processors)  # Many Linux machines: physical = threads unless hyperthreaded

        freq_match = re.search(r'cpu MHz\s+: ([\d.]+)', cpuinfo)
        freq = float(freq_match.group(1)) / 1000 if freq_match else 0

        return f"{model} ({physical_cores}C/{threads}T @ {freq:.1f} GHz)"
    except Exception as e:
        exception(session_state_get('current_console_log_key'), 
                  f"failed to extract CPU info: {e}")
        return "Information not available"

def get_gpu_info_fallback(rocminfo_data: str):
    # Split agents - every agent block starts with '*******' at line start
    # This regex will capture all agent blocks.
    agent_blocks = re.split(r'^\*{7,}\s*$', rocminfo_data, flags=re.MULTILINE)

    gpu_info = []
    for block in agent_blocks:
        # Make sure we're looking at a block with 'Agent' and 'Device Type: GPU'
        if 'Device Type:' in block:
            if re.search(r'Device Type:\s*GPU', block):
                # Try Marketing Name, fallback to Name
                m = re.search(r'Marketing Name:\s*([^\n]+)', block)
                marketing_name = m.group(1).strip() if m else None
                if not marketing_name:
                    n = re.search(r'Name:\s*([^\n]+)', block)
                    marketing_name = n.group(1).strip() if n else "Unknown"
                # gfx code: look for first 'gfxNNN'
                gfx = None
                g = re.search(r'\bgfx\d+\b', block)
                if g:
                    gfx = g.group(0)
                # Save info
                gpu_info.append((marketing_name, gfx))

    if not gpu_info:
        warning(session_state_get('current_console_log_key'), 
                f"failed to list system GPUs")
        return "Information not available"

    # Count and summarize
    counter = Counter(gpu_info)
    info_strings = [f"{count} x {model} ({gfx})" for (model, gfx), count in counter.items()]
    return ", ".join(info_strings)

def get_gpu_info():
    """
    Extracts GPU names and architectures using ROCm's rocminfo tool.
    Returns a formatted string or 'Information not available' on error.
    """
    try:
        output = subprocess.check_output(['rocminfo'], text=True)
        lines = output.splitlines()

        gpus = []
        current_gpu = None
        for line in lines:
            marketing = re.search(r'Marketing Name:\s*(.+)', line)
            name = re.search(r'Name:\s*(gfx\d+)', line)
            if marketing:
                current_gpu = {'marketing_name': marketing.group(1), 'arch': None}
            if current_gpu is not None and name:
                current_gpu['arch'] = name.group(1)
                if current_gpu['marketing_name'].strip().startswith("AMD Instinct"):
                    gpus.append(current_gpu)
                current_gpu = None

        if not gpus:
            # Try alternate mechanism to fetch GPUs
            return get_gpu_info_fallback(output)

        tuples = [(g['marketing_name'].strip(), g['arch']) for g in gpus if g['arch']]
        counter = Counter(tuples)
        info_strings = [
            f"{count} x {gpu} ({arch})"
            for (gpu, arch), count in counter.items()
        ]
        return ", ".join(info_strings)
    except Exception as e:
        return "Information not available"

def get_package_version(pkg_name):
    """
    Returns the installed version of a Python package, or 'Information not available'.
    """
    try:
        version = importlib.metadata.version(pkg_name)
    except Exception as e:
        warning(session_state_get('current_console_log_key'), 
                f"failed to get package {pkg_name} version: {e}")
        version = "Information not available"
    return version

# Convert a DLDataType to a familiar dtype string
def dl_datatype_to_str(dl_dtype):
    code_map = {0: 'int', 1: 'uint', 2: 'float'}
    base = code_map.get(dl_dtype.code, 'unknown')
    bits = dl_dtype.bits
    lanes = dl_dtype.lanes
    if lanes > 1:
        return f"{base}{bits}x{lanes}"
    else:
        return f"{base}{bits}"

# Mapping from full unit names to abbreviations/symbols
unit_map = {
    'micrometer': 'µm',
    'millimeter': 'mm',
    'centimeter': 'cm',
    'meter': 'm',
    'nanometer': 'nm',
    'pixel': 'px',
    'color': '',  # Not a spatial unit
}

def filesize_h(filename):
    """
    Returns the human-readable file size for the given path.
    """
    try:
        size = os.path.getsize(filename)
        for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return "%3.1f %s" % (size, unit)
            size /= 1024.0
        return "%3.1f %s" % (size, 'PB')
    except Exception as e:
        exception(session_state_get('current_console_log_key'), 
                  f"failed to retrieve human-friendly filesize: {e}")
        return "Information not available"

def get_image_shape(path):
    import pyvips
    image = pyvips.Image.new_from_file(path, access="sequential")
    return image.width, image.height

# Helper to filter out non-NIfTI files 
def is_nii(filename):
    return filename.endswith(".nii") or filename.endswith(".nii.gz")

