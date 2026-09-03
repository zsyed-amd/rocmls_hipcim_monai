"""System info: GPU / CPU / key package versions. Wraps components.utility."""

from fastapi import APIRouter

from backend.schemas import SystemInfo
from components import utility

router = APIRouter(prefix="/api/system", tags=["system"])

_TRACKED_PACKAGES = ("torch", "monai", "cucim", "cupy", "transformers", "streamlit")


@router.get("/info", response_model=SystemInfo)
def system_info():
    versions = {pkg: utility.get_package_version(pkg) for pkg in _TRACKED_PACKAGES}
    return SystemInfo(
        gpu=utility.get_gpu_info(),
        cpu=utility.get_cpu_info(),
        versions=versions,
    )
