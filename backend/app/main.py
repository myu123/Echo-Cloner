"""
FastAPI main application entry point
"""
import logging
import sys
from typing import Tuple

import torch
import torchaudio
import soundfile as sf

# ---------------------------------------------------------------------------
# Patch torchaudio.load globally.  torchaudio >=2.9 delegates to torchcodec
# which is not installed.  We fall back to soundfile for WAV decoding.
# This MUST run before any TTS / model code imports torchaudio.
# ---------------------------------------------------------------------------
_orig_ta_load = torchaudio.load


def _sf_torchaudio_load(
    uri, frame_offset=0, num_frames=-1, normalize=True,
    channels_first=True, format=None, buffer_size=4096, backend=None,
) -> Tuple[torch.Tensor, int]:
    try:
        return _orig_ta_load(
            uri, frame_offset=frame_offset, num_frames=num_frames,
            normalize=normalize, channels_first=channels_first,
            format=format, buffer_size=buffer_size, backend=backend,
        )
    except (ImportError, RuntimeError):
        pass
    data, sample_rate = sf.read(str(uri), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
    if frame_offset > 0:
        waveform = waveform[:, frame_offset:]
    if num_frames > 0:
        waveform = waveform[:, :num_frames]
    if not channels_first:
        waveform = waveform.transpose(0, 1)
    return waveform, sample_rate


torchaudio.load = _sf_torchaudio_load
# ---------------------------------------------------------------------------

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from .models.config import settings, ensure_directories
from .api.routes import upload, transcribe, train, generate, models, system
from .api.websocket import websocket_endpoint
from .utils.cuda_utils import check_cuda_available, get_gpu_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Echo Cloner API",
    description="Echo Cloner - XTTS-v2 powered voice cloning and synthesis",
    version="1.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("=" * 60)
    print("Echo Cloner API Starting...")
    print("=" * 60)

    # Ensure all directories exist
    ensure_directories()
    print("✓ Directories initialized")

    # Check CUDA availability
    cuda_available = check_cuda_available()
    if cuda_available:
        gpu_info = get_gpu_info()
        print(f"✓ CUDA available: {gpu_info.device_name}")
        print(f"  - CUDA Version: {gpu_info.cuda_version}")
        print(f"  - GPU Memory: {gpu_info.memory_total:.2f} GB")
    else:
        print("⚠ WARNING: CUDA not available - training will not work!")
        print("  Please ensure PyTorch with CUDA support is installed")

    print("=" * 60)
    print(f"API running on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Docs available at http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Echo Cloner API...")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Echo Cloner API",
        "version": "1.0.0",
        "docs": "/docs",
        "cuda_available": check_cuda_available()
    }


# Include routers
app.include_router(upload.router)
app.include_router(transcribe.router)
app.include_router(train.router)
app.include_router(generate.router)
app.include_router(models.router)
app.include_router(system.router)


# WebSocket endpoint
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for training updates"""
    await websocket_endpoint(websocket)


# Run server
def run():
    """Run the API server"""
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )


if __name__ == "__main__":
    run()
