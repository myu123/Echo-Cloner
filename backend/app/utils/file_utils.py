"""
File and directory utility functions
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import hashlib
from datetime import datetime
from ..models.schemas import DiskInfo, ModelMetadata


def get_file_hash(file_path: Path) -> str:
    """Generate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]  # Use first 16 chars


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    return file_path.stat().st_size if file_path.exists() else 0


def get_disk_info(path: Path) -> DiskInfo:
    """Get disk space information for a given path"""
    try:
        stat = shutil.disk_usage(path)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        percent_used = (stat.used / stat.total * 100) if stat.total > 0 else 0

        return DiskInfo(
            total=round(total_gb, 2),
            used=round(used_gb, 2),
            free=round(free_gb, 2),
            percent_used=round(percent_used, 2)
        )
    except Exception as e:
        print(f"Error getting disk info: {e}")
        return DiskInfo(total=0, used=0, free=0, percent_used=0)


def ensure_directory(directory: Path) -> None:
    """Create directory if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def delete_directory(directory: Path) -> bool:
    """Delete directory and all its contents"""
    try:
        if directory.exists() and directory.is_dir():
            shutil.rmtree(directory)
            return True
        return False
    except Exception as e:
        print(f"Error deleting directory {directory}: {e}")
        return False


def save_json(data: dict, file_path: Path) -> bool:
    """Save dictionary as JSON file"""
    try:
        ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: Path) -> Optional[dict]:
    """Load JSON file as dictionary"""
    try:
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None


def save_model_metadata(model_dir: Path, metadata: ModelMetadata) -> bool:
    """Save model metadata to JSON file"""
    metadata_path = model_dir / "metadata.json"
    metadata_dict = metadata.dict()
    return save_json(metadata_dict, metadata_path)


def load_model_metadata(model_dir: Path) -> Optional[ModelMetadata]:
    """Load model metadata from JSON file"""
    metadata_path = model_dir / "metadata.json"
    data = load_json(metadata_path)
    if data:
        try:
            return ModelMetadata(**data)
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            return None
    return None


def get_all_models(models_dir: Path) -> List[ModelMetadata]:
    """Get metadata for all models in the models directory"""
    models = []

    if not models_dir.exists():
        return models

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            metadata = load_model_metadata(model_dir)
            if metadata:
                models.append(metadata)

    # Sort by creation date, newest first
    models.sort(key=lambda x: x.created_at, reverse=True)
    return models


def delete_model(model_dir: Path) -> bool:
    """Delete a model directory and all its contents"""
    return delete_directory(model_dir)


def get_audio_files(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Get all audio files in a directory"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']

    audio_files = []
    if directory.exists():
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(audio_files)


def generate_unique_id() -> str:
    """Generate a unique ID based on timestamp and random hash"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.sha256(os.urandom(32)).hexdigest()[:8]
    return f"{timestamp}_{random_hash}"


def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters"""
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    return f"{name}{ext}"
