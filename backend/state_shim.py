"""Install a minimal fake `streamlit` module so the demo's `components`/`products`
code (which reads `st.session_state` and calls `st.*` widgets at import/runtime)
can be imported and driven from a non-Streamlit process (our FastAPI backend).

IMPORT THIS BEFORE ANY `components.*` OR `products.*` IMPORT.

`st.session_state` becomes a process-global dict (the demo is single-session, so a
shared dict matches Streamlit's own model). Every other `st.*` attribute resolves
to a universal no-op stub that survives being called, indexed, used as a context
manager, or used as a decorator — enough to import the UI modules without executing
their Streamlit rendering (which we never call from the API path)."""

import sys
import types


class SessionState(dict):
    """dict that also supports attribute access, like Streamlit's session_state."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class _Stub:
    """Universal no-op: callable, indexable, iterable, context manager, decorator."""

    def __call__(self, *args, **kwargs):
        # Support bare decorator use like @st.cache_data
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __getattr__(self, _name):
        return self

    def __getitem__(self, _key):
        return self

    def __iter__(self):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __bool__(self):
        return False


_SESSION_STATE = SessionState()
_STUB = _Stub()


def get_session_state():
    """Return the process-global session-state dict shared with the demo code."""
    return _SESSION_STATE


def install():
    """Idempotently install the fake `streamlit` module into sys.modules."""
    existing = sys.modules.get("streamlit")
    if isinstance(existing, types.ModuleType) and getattr(existing, "_amd_fake", False):
        return _SESSION_STATE

    module = types.ModuleType("streamlit")
    module._amd_fake = True
    module.session_state = _SESSION_STATE

    def _module_getattr(name):
        if name == "session_state":
            return _SESSION_STATE
        return _STUB

    module.__getattr__ = _module_getattr
    sys.modules["streamlit"] = module
    return _SESSION_STATE


# Install on import so a simple `import backend.state_shim` is enough.
install()
