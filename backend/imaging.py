"""Image encoding helpers shared across routers.

Force the matplotlib Agg backend at import time so any figure building the demo does
in a worker thread never tries to touch a display."""

import base64
import io

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image


def _to_pil(image):
    """Coerce a PIL image or HxW / HxWxC numpy array into a PIL image."""
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        amin, amax = float(arr.min()), float(arr.max())
        if amax > amin:
            arr = (arr - amin) / (amax - amin) * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L")
    return Image.fromarray(arr)


def to_png_bytes(image):
    """Return raw PNG bytes for a PIL image or numpy array."""
    buf = io.BytesIO()
    _to_pil(image).save(buf, format="PNG")
    return buf.getvalue()


def to_base64_png(image):
    """Return a data-URI base64 PNG string for a PIL image or numpy array."""
    encoded = base64.b64encode(to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
