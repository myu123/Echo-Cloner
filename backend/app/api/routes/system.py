"""
System information and health check API routes
"""
from fastapi import APIRouter, HTTPException
import sys
import torch
from ...models.schemas import SystemInfo, HealthResponse, GPUInfo
from ...utils.cuda_utils import get_gpu_info, check_cuda_available, get_cpu_memory_info
from ...utils.file_utils import get_disk_info
from ...models.config import settings


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status
    """
    return HealthResponse(
        status="healthy",
        cuda_available=check_cuda_available()
    )


@router.get("/cuda", response_model=GPUInfo)
async def get_cuda_info():
    """
    Get CUDA/GPU information
    Returns GPU details if available
    """
    try:
        gpu_info = get_gpu_info()
        return gpu_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    Get comprehensive system information
    Includes GPU, disk, Python, and PyTorch versions
    """
    try:
        # Get GPU info
        gpu_info = get_gpu_info()

        # Get disk info
        disk_info = get_disk_info(settings.BASE_DIR)

        # Get versions
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        torch_version = torch.__version__

        return SystemInfo(
            cuda_available=check_cuda_available(),
            gpu_info=gpu_info if gpu_info.available else None,
            disk_info=disk_info,
            python_version=python_version,
            torch_version=torch_version
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory")
async def get_memory_info():
    """
    Get current memory usage (CPU and GPU)
    """
    try:
        # CPU memory
        cpu_memory = get_cpu_memory_info()

        # GPU memory
        gpu_memory = None
        if check_cuda_available():
            gpu_info = get_gpu_info()
            if gpu_info.available:
                gpu_memory = {
                    "total": gpu_info.memory_total,
                    "allocated": gpu_info.memory_allocated,
                    "reserved": gpu_info.memory_reserved,
                    "free": gpu_info.memory_total - gpu_info.memory_allocated if gpu_info.memory_total else None
                }

        return {
            "cpu_memory": cpu_memory,
            "gpu_memory": gpu_memory
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage")
async def get_storage_info():
    """
    Get storage usage information
    """
    try:
        from pathlib import Path

        disk_info = get_disk_info(settings.BASE_DIR)

        # Get size of key directories
        def get_dir_size(directory: Path) -> float:
            """Get directory size in GB"""
            if not directory.exists():
                return 0.0

            total = 0
            for item in directory.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size

            return total / (1024**3)  # Convert to GB

        uploads_size = get_dir_size(settings.UPLOADS_DIR)
        processed_size = get_dir_size(settings.PROCESSED_DIR)
        models_size = get_dir_size(settings.MODELS_DIR)
        cache_size = get_dir_size(settings.CACHE_DIR)

        return {
            "disk": disk_info.dict(),
            "directories": {
                "uploads": round(uploads_size, 2),
                "processed": round(processed_size, 2),
                "models": round(models_size, 2),
                "cache": round(cache_size, 2),
                "total_used": round(uploads_size + processed_size + models_size + cache_size, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_cache():
    """
    Clear CUDA cache and temporary files
    """
    try:
        from ...utils.cuda_utils import clear_cuda_cache

        # Clear CUDA cache
        if check_cuda_available():
            clear_cuda_cache()

        return {
            "success": True,
            "message": "Cache cleared successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
