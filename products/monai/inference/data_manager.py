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
import sys
import subprocess
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from components.diagnostics import info, warning, error
from products.monai.console import INFER_CONSOLE_LOG_KEY

# Import CI utilities for enhanced bundle management
CI_PATH = "model-zoo/ci"  # CI utilities from git cloned model-zoo
# Add absolute path for CI utilities
import os
from pathlib import Path

# Get project root directory
current_file = Path(__file__).resolve()
project_root = None
for parent in current_file.parents:
    if parent.name == "rocm-ls-examples":
        project_root = parent
        break

if project_root is None:
    # Fallback - assume we're in the right place
    project_root = Path(__file__).parent.parent.parent.parent

ci_absolute_path = project_root / CI_PATH
sys.path.append(str(ci_absolute_path))

try:
    from utils import (
        download_large_files,
        get_latest_version,
        get_json_dict
    )
    from download_latest_bundle import download_latest_bundle
    CI_UTILS_AVAILABLE = True
except ImportError as e:
    CI_UTILS_AVAILABLE = False

# MONAI Model Zoo paths:
# - model-zoo/: Git cloned repository with CI utilities and model definitions
# - demo/model_zoo_models/: Actual downloaded model files for demo
MODEL_ZOO_PATH = "demo/model_zoo_models"

# Convert to absolute path for existence check
model_zoo_absolute = project_root / MODEL_ZOO_PATH

def get_bundle_path(model_id: str) -> str:
    """Get the bundle path for a model"""
    # Map model IDs to their actual bundle directories
    bundle_map = {
        "spleen_ct_seg": "spleen_ct_segmentation",
        "pathology_tumor_detection": "pathology_tumor_detection"
    }

    bundle_name = bundle_map.get(model_id, model_id)
    # Return absolute path
    return str(project_root / MODEL_ZOO_PATH / bundle_name)

def read_large_files_config(model_id: str) -> List[Dict]:
    """Read the large_files.yml configuration for a model"""
    bundle_path = get_bundle_path(model_id)
    large_files_path = os.path.join(bundle_path, "large_files.yml")

    if not os.path.exists(large_files_path):
        error(INFER_CONSOLE_LOG_KEY, f"No large_files.yml found for {model_id}")
        return []

    try:
        with open(large_files_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('large_files', [])
    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"Failed to read large_files.yml: {str(e)}")
        return []

def download_using_ci_utilities(model_id: str, progress_callback=None) -> bool:
    """Use CI utilities for enhanced bundle downloading"""
    if not CI_UTILS_AVAILABLE:
        return download_using_monai_bundle_fallback(model_id, progress_callback)

    bundle_path = get_bundle_path(model_id)

    if not os.path.exists(bundle_path):
        error(INFER_CONSOLE_LOG_KEY, f"Bundle not found: {bundle_path}")
        return False

    try:
        info(INFER_CONSOLE_LOG_KEY, f"Using CI utilities to download {model_id}...")

        if progress_callback:
            progress_callback("Getting latest version...", 0.1)

        # Get the model info to find latest version
        model_info_path = os.path.join(MODEL_ZOO_PATH, "model_info.json")
        if os.path.exists(model_info_path):
            try:
                bundle_name = get_bundle_name_from_id(model_id)
                version = get_latest_version(bundle_name, model_info_path)
                info(INFER_CONSOLE_LOG_KEY, f"Latest version for {bundle_name}: {version}")

                if progress_callback:
                    progress_callback("Downloading latest bundle...", 0.3)

                # Use the CI download function
                download_latest_bundle(bundle_name, MODEL_ZOO_PATH, bundle_path)

            except Exception as e:
                warning(INFER_CONSOLE_LOG_KEY, f"Version lookup failed: {e}")

        if progress_callback:
            progress_callback("Downloading large files...", 0.6)

        # Download large files using CI utilities
        download_large_files(bundle_path, "large_files.yml")

        if progress_callback:
            progress_callback("Download completed", 1.0)

        info(INFER_CONSOLE_LOG_KEY, f"Successfully downloaded {model_id} using CI utilities")
        return True

    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"CI download failed: {str(e)}")
        # Fallback to original method
        return download_using_monai_bundle_fallback(model_id, progress_callback)

def get_bundle_name_from_id(model_id: str) -> str:
    """Convert model ID to bundle name"""
    bundle_map = {
        "spleen_ct_seg": "spleen_ct_segmentation",
        "pathology_tumor_detection": "pathology_tumor_detection"
    }
    return bundle_map.get(model_id, model_id)

def download_using_monai_bundle_fallback(model_id: str, progress_callback=None) -> bool:
    """Fallback method: Use MONAI bundle download command to get model and data"""
    bundle_path = get_bundle_path(model_id)

    if not os.path.exists(bundle_path):
        error(INFER_CONSOLE_LOG_KEY, f"Bundle not found: {bundle_path}")
        return False

    try:
        info(INFER_CONSOLE_LOG_KEY, f"Downloading resources for {model_id} (fallback method)...")

        if progress_callback:
            progress_callback("Downloading model files...", 0.5)

        # Use MONAI bundle download command with explicit bundle path
        result = subprocess.run([
            "python", "-m", "monai.bundle", "download",
            "--name", bundle_path,
            "--version", "latest"
        ], capture_output=True, text=True, timeout=300, cwd=os.getcwd())

        if result.returncode == 0:
            info(INFER_CONSOLE_LOG_KEY, "Download completed successfully")
            info(INFER_CONSOLE_LOG_KEY, f"stdout: {result.stdout}")
            return True
        else:
            error(INFER_CONSOLE_LOG_KEY, f"Download failed: {result.stderr}")
            # Try alternative approach
            return download_large_files_directly(model_id, progress_callback)

    except subprocess.TimeoutExpired:
        error(INFER_CONSOLE_LOG_KEY, "Download timed out")
        return False
    except Exception as e:
        error(INFER_CONSOLE_LOG_KEY, f"Download error: {str(e)}")
        # Try alternative approach
        return download_large_files_directly(model_id, progress_callback)

def download_large_files_directly(model_id: str, progress_callback=None) -> bool:
    """Download files directly using the large_files.yml configuration"""

    large_files = read_large_files_config(model_id)
    if not large_files:
        error(INFER_CONSOLE_LOG_KEY, f"No large files configuration found for {model_id}")
        return False

    bundle_path = get_bundle_path(model_id)
    success_count = 0

    for i, file_info in enumerate(large_files):
        file_path = os.path.join(bundle_path, file_info['path'])
        file_url = file_info['url']

        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Skip if file already exists
        if os.path.exists(file_path):
            info(INFER_CONSOLE_LOG_KEY, f"File already exists: {file_info['path']}")
            success_count += 1
            continue

        if progress_callback:
            progress_callback(f"Downloading {file_info['path']}...", i / len(large_files))

        info(INFER_CONSOLE_LOG_KEY, f"Downloading {file_info['path']} from {file_url}")

        try:
            response = requests.get(file_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            info(INFER_CONSOLE_LOG_KEY, f"Successfully downloaded: {file_info['path']}")
            success_count += 1

        except Exception as e:
            error(INFER_CONSOLE_LOG_KEY, f"Failed to download {file_info['path']}: {str(e)}")

    return success_count > 0

def download_sample_data(model_id: str, progress_callback=None) -> List[str]:
    """Download sample data using enhanced CI utilities with fallback"""
    info(INFER_CONSOLE_LOG_KEY, f"Downloading sample data for {model_id} using enhanced CI utilities")

    # Try CI utilities first, then fallback to original method
    if download_using_ci_utilities(model_id, progress_callback):
        # Return list of available data files after download
        return get_available_data_files(model_id)
    else:
        error(INFER_CONSOLE_LOG_KEY, f"Failed to download data for {model_id}")
        return []

def download_model(model_id: str, progress_callback=None) -> Optional[str]:
    """Download pre-trained model using enhanced CI utilities with fallback"""
    info(INFER_CONSOLE_LOG_KEY, f"Downloading model for {model_id} using enhanced CI utilities")

    # Try CI utilities first, then fallback to original method
    if download_using_ci_utilities(model_id, progress_callback):
        # Return the model file path
        model_files = get_available_model_files(model_id)
        return model_files[0] if model_files else None
    else:
        error(INFER_CONSOLE_LOG_KEY, f"Failed to download model for {model_id}")
        return None

def get_available_data_files(model_id: str) -> List[str]:
    """Get list of available data files by simply listing demo/data directory"""
    data_files = []

    # Map model IDs to their data directories
    model_data_dirs = {
        'spleen_ct_seg': 'spleen_ct_segmentation',
        'pathology_tumor_detection': 'pathology_tumor_detection'
    }

    data_dir_name = model_data_dirs.get(model_id, model_id)
    data_dir = project_root / "demo/data" / data_dir_name

    if data_dir.exists():
        # Return relative paths from project root
        for file_path in data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                # Skip CSV files - we want actual image data
                if file_path.suffix.lower() in ['.csv']:
                    continue
                # Convert to relative path
                relative_path = file_path.relative_to(project_root)
                data_files.append(str(relative_path))

    # For pathology models, also check sample_images/camelyon directory
    if model_id == 'pathology_tumor_detection':
        camelyon_dir = project_root / "sample_images" / "camelyon"
        if camelyon_dir.exists():
            for file_path in camelyon_dir.iterdir():
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Only include .tif/.tiff files for pathology
                    if file_path.suffix.lower() in ['.tif', '.tiff']:
                        relative_path = file_path.relative_to(project_root)
                        data_files.append(str(relative_path))

    return sorted(data_files)

def get_available_model_files(model_id: str) -> List[str]:
    """Get list of available model files by listing custom_trained directory"""
    model_files = []

    # Check in demo/custom_trained for the model
    custom_trained_dir = project_root / "demo/custom_trained"

    if custom_trained_dir.exists():
        # List all subdirectories (model folders)
        for model_folder in custom_trained_dir.iterdir():
            if model_folder.is_dir():
                # Only check folders that match the model_id
                if model_folder.name != model_id:
                    continue
                    
                # Look for models subdirectory
                models_dir = model_folder / "models"
                if models_dir.exists():
                    # List all .pt and .pth files
                    for model_file in models_dir.iterdir():
                        if model_file.is_file() and model_file.suffix in ['.pt', '.pth']:
                            # Convert to relative path
                            relative_path = model_file.relative_to(project_root)
                            model_files.append(str(relative_path))

    return sorted(model_files)

def get_available_model_folders() -> List[str]:
    """Get list of available model folders in custom_trained directory"""
    model_folders = []
    
    custom_trained_dir = project_root / "demo/custom_trained"
    
    if custom_trained_dir.exists():
        for folder in custom_trained_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('.'):
                # Check if it has a models subdirectory
                if (folder / "models").exists():
                    model_folders.append(folder.name)
    
    return sorted(model_folders)

def get_model_files_for_folder(model_folder_name: str) -> List[str]:
    """Get list of model files in a specific model folder"""
    model_files = []

    models_dir = project_root / "demo/custom_trained" / model_folder_name / "models"

    if models_dir.exists():
        for model_file in models_dir.iterdir():
            if model_file.is_file() and model_file.suffix in ['.pt', '.pth']:
                # Convert to relative path
                relative_path = model_file.relative_to(project_root)
                model_files.append(str(relative_path))

    return sorted(model_files)

def get_data_info(model_id: str) -> Dict:
    """Get information about available data for a model"""
    data_files = get_available_data_files(model_id)
    model_files = get_available_model_files(model_id)

    return {
        "model_id": model_id,
        "data_available": len(data_files) > 0,
        "model_available": len(model_files) > 0,
        "data_files": data_files,
        "model_files": model_files,
        "data_count": len(data_files),
        "model_count": len(model_files)
    }

def is_data_available(model_id: str) -> bool:
    """Check if data is available for a model"""
    return len(get_available_data_files(model_id)) > 0

def is_model_available(model_id: str) -> bool:
    """Check if model is available for inference"""
    return len(get_available_model_files(model_id)) > 0
