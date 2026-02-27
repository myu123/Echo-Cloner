"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False  # Disable auto-reload to avoid trainer log file conflicts

    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    DATASETS_DIR: Path = DATA_DIR / "datasets"
    MODELS_DIR: Path = BASE_DIR / "trained_models"
    CACHE_DIR: Path = BASE_DIR / "cache"

    # Audio Processing Settings
    SAMPLE_RATE: int = 22050
    TARGET_SEGMENT_LENGTH_MIN: float = 3.0  # seconds
    TARGET_SEGMENT_LENGTH_MAX: float = 10.0  # seconds
    MIN_TOTAL_AUDIO_DURATION: float = 600.0  # 10 minutes in seconds
    RECOMMENDED_MIN_AUDIO_DURATION: float = 1200.0  # 20 minutes
    MAX_AUDIO_DURATION: float = 7200.0  # 2 hours
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB in bytes
    SUPPORTED_FORMATS: list = ["mp3", "wav", "m4a", "ogg", "flac", "aac"]

    # Whisper Settings
    WHISPER_MODEL: str = "large-v2"
    WHISPER_DEVICE: str = "cuda"  # Will fallback to cpu if not available
    WHISPER_COMPUTE_TYPE: str = "float16"  # Use fp16 for faster inference on GPU
    USE_STABLE_TS: bool = True  # Prefer stable-ts (Whisper with VAD + better timestamps) when available
    STABLE_TS_VAD: bool = True  # Enable built-in VAD in stable-ts for cleaner segments

    # XTTS Training Settings
    XTTS_BASE_MODEL_DIR: Optional[Path] = None  # If None, we try common cache locations
    DEFAULT_EPOCHS: int = 15
    DEFAULT_BATCH_SIZE: int = 2
    DEFAULT_LEARNING_RATE: float = 5e-6
    DEFAULT_GRADIENT_ACCUMULATION: int = 2
    DEFAULT_SAVE_STEP: int = 1000
    MIN_SEGMENTS_FOR_TRAINING: int = 50

    # System
    MAX_WORKERS: int = 4
    WEBSOCKET_PING_INTERVAL: int = 30
    WEBSOCKET_PING_TIMEOUT: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Training configuration presets
TRAINING_PRESETS = {
    "fast": {
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 1,
        "description": "Quick training, lower quality"
    },
    "balanced": {
        "epochs": 15,
        "batch_size": 2,
        "learning_rate": 5e-6,
        "gradient_accumulation_steps": 2,
        "description": "Recommended for most cases"
    },
    "high_quality": {
        "epochs": 25,
        "batch_size": 1,
        "learning_rate": 2e-6,
        "gradient_accumulation_steps": 4,
        "description": "Best quality, slower training"
    }
}


def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        settings.DATA_DIR,
        settings.UPLOADS_DIR,
        settings.PROCESSED_DIR,
        settings.DATASETS_DIR,
        settings.MODELS_DIR,
        settings.CACHE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
