"""FastAPI backend for the standalone MONAI / hipCIM demo.

Run from the repo root inside the ROCm py3.10 container:
    uvicorn backend.app:app --host 0.0.0.0 --port 8600 --workers 1

The import order matters: the fake `streamlit` shim and the console-log patch must be
installed before any `components.*` / `products.*` module is imported by the routers."""

import backend.state_shim  # noqa: F401  installs fake streamlit (must be first)
from backend import log_bus

log_bus.install()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import system

app = FastAPI(title="AMD Instinct ROCm-LS Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
