"""Capture the demo's console logging and route it to per-job queues for SSE.

Every diagnostic in the demo funnels through `components.console_log.append_console_log`
(via `components.diagnostics.{info,warning,error,exception}` and a few direct callers).
We monkeypatch that function everywhere it was imported-by-value, so a running job's
log lines land on a thread-local `queue.Queue` that an SSE endpoint can drain.

Modules that do `from components.console_log import append_console_log` capture the
reference at import time, so patching only the source module is not enough — we rebind
the name in each importing module too."""

import importlib
import threading

import backend.state_shim  # noqa: F401  (ensure fake streamlit before component imports)
from components import console_log

_local = threading.local()

# Modules known to import append_console_log by value. Heavy ones (transformers/monai)
# are patched lazily via patch_module() when their phase loads.
_EAGER_MODULES = ("components.diagnostics",)

_original_append = console_log.append_console_log


def set_active_queue(q):
    _local.queue = q


def clear_active_queue():
    _local.queue = None


def _active_queue():
    return getattr(_local, "queue", None)


def _patched_append(console_log_key, line):
    # Preserve original behaviour (session_state buffer + print).
    try:
        _original_append(console_log_key, line)
    except Exception:
        pass
    q = _active_queue()
    if q is not None:
        try:
            q.put(("log", console_log.strip_span_tags(line)))
        except Exception:
            pass


def patch_module(dotted_name):
    """Rebind append_console_log in an already-imported (or importable) module."""
    try:
        mod = importlib.import_module(dotted_name)
    except Exception:
        return
    if hasattr(mod, "append_console_log"):
        mod.append_console_log = _patched_append


def install():
    """Patch the source and all eager importers. Call once at startup."""
    console_log.append_console_log = _patched_append
    for name in _EAGER_MODULES:
        patch_module(name)
