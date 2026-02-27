"""
CUDA and GPU utility functions
"""
import torch
import psutil
from typing import Optional, Dict
from ..models.schemas import GPUInfo


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def get_gpu_info() -> GPUInfo:
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            device_count=0
        )

    try:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else None
        cuda_version = torch.version.cuda

        # Get memory info for the first GPU
        memory_total = None
        memory_allocated = None
        memory_reserved = None

        if device_count > 0:
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)

        return GPUInfo(
            available=True,
            device_name=device_name,
            cuda_version=cuda_version,
            device_count=device_count,
            memory_total=memory_total,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved
        )
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return GPUInfo(available=False, device_count=0)


def get_gpu_utilization() -> Dict[str, Optional[float]]:
    """
    Get current GPU utilization and memory usage
    Returns dict with gpu_utilization (%), gpu_memory_used (GB), gpu_memory_total (GB)
    """
    if not torch.cuda.is_available():
        return {
            "gpu_utilization": None,
            "gpu_memory_used": None,
            "gpu_memory_total": None
        }

    try:
        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        # Calculate utilization percentage
        gpu_utilization = (memory_allocated / memory_total * 100) if memory_total > 0 else 0

        return {
            "gpu_utilization": round(gpu_utilization, 2),
            "gpu_memory_used": round(memory_allocated, 2),
            "gpu_memory_total": round(memory_total, 2)
        }
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return {
            "gpu_utilization": None,
            "gpu_memory_used": None,
            "gpu_memory_total": None
        }


def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_device() -> str:
    """Get the optimal device (cuda or cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_cpu_memory_info() -> Dict[str, float]:
    """Get CPU memory information in GB"""
    memory = psutil.virtual_memory()
    return {
        "total": round(memory.total / (1024**3), 2),
        "available": round(memory.available / (1024**3), 2),
        "used": round(memory.used / (1024**3), 2),
        "percent": memory.percent
    }
