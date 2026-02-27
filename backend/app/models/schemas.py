"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# Upload schemas
class AudioFileInfo(BaseModel):
    filename: str
    size: int  # bytes
    duration: Optional[float] = None  # seconds
    format: str
    uploaded_at: datetime = Field(default_factory=datetime.now)


class UploadResponse(BaseModel):
    success: bool
    file_id: str
    message: str
    file_info: Optional[AudioFileInfo] = None


# Transcription schemas
class AudioSegment(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    text: str
    audio_path: str


class TranscriptionRequest(BaseModel):
    file_ids: List[str]
    language: Optional[str] = "en"


class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    total_segments: int
    total_duration: float  # seconds
    segments: List[AudioSegment]


# Training schemas
class TrainingConfig(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    epochs: int = Field(default=15, ge=5, le=30)
    batch_size: int = Field(default=2, ge=1, le=4)
    learning_rate: float = Field(default=5e-6, gt=0)
    gradient_accumulation_steps: int = Field(default=2, ge=1, le=8)
    save_step: int = Field(default=1000, ge=100)

    @validator('model_name')
    def validate_model_name(cls, v):
        # Only allow alphanumeric, spaces, hyphens, and underscores
        import re
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Model name can only contain letters, numbers, spaces, hyphens, and underscores')
        return v.strip()


class TrainingRequest(BaseModel):
    config: TrainingConfig
    dataset_path: Optional[str] = None


class TrainingProgress(BaseModel):
    status: TrainingStatus
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    time_elapsed: float = 0.0  # seconds
    time_remaining: Optional[float] = None  # seconds
    gpu_utilization: Optional[float] = None  # percentage
    gpu_memory_used: Optional[float] = None  # GB
    gpu_memory_total: Optional[float] = None  # GB
    message: Optional[str] = None


class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    training_started: bool = False


# Inference schemas
class GenerateRequest(BaseModel):
    model_id: str
    text: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="en")
    temperature: float = Field(default=0.65, ge=0.1, le=1.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=20.0)
    length_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    top_k: int = Field(default=50, ge=1, le=100)
    top_p: float = Field(default=0.85, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    success: bool
    message: str
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    duration: Optional[float] = None


# Model management schemas
class ModelMetadata(BaseModel):
    model_id: str
    model_name: str
    created_at: datetime
    training_duration: float  # seconds
    total_audio_duration: float  # seconds
    num_clips: int
    epochs: int
    batch_size: int
    learning_rate: float
    final_loss: Optional[float] = None
    file_size: int  # bytes
    quality_score: Optional[float] = None  # 0-100


class ModelInfo(BaseModel):
    metadata: ModelMetadata
    training_logs: Optional[str] = None


class ModelsListResponse(BaseModel):
    success: bool
    models: List[ModelMetadata]
    total_count: int


class ModelUpdateRequest(BaseModel):
    model_name: Optional[str] = None


class ModelDeleteResponse(BaseModel):
    success: bool
    message: str


# System schemas
class GPUInfo(BaseModel):
    available: bool
    device_name: Optional[str] = None
    cuda_version: Optional[str] = None
    device_count: int = 0
    memory_total: Optional[float] = None  # GB
    memory_allocated: Optional[float] = None  # GB
    memory_reserved: Optional[float] = None  # GB


class DiskInfo(BaseModel):
    total: float  # GB
    used: float  # GB
    free: float  # GB
    percent_used: float


class SystemInfo(BaseModel):
    cuda_available: bool
    gpu_info: Optional[GPUInfo] = None
    disk_info: DiskInfo
    python_version: str
    torch_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    cuda_available: bool
