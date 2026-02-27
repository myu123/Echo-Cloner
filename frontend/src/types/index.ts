/**
 * TypeScript types and interfaces for the Echo Cloner app
 * Matches backend Pydantic schemas
 */

export enum TrainingStatus {
  IDLE = 'idle',
  PROCESSING = 'processing',
  TRAINING = 'training',
  COMPLETED = 'completed',
  FAILED = 'failed',
  STOPPED = 'stopped',
}

// Audio File Types
export interface AudioFileInfo {
  filename: string;
  size: number;
  duration?: number;
  format: string;
  uploaded_at: string;
}

export interface UploadResponse {
  success: boolean;
  file_id: string;
  message: string;
  file_info?: AudioFileInfo;
}

// Transcription Types
export interface AudioSegment {
  segment_id: string;
  start_time: number;
  end_time: number;
  duration: number;
  text: string;
  audio_path: string;
}

export interface TranscriptionRequest {
  file_ids: string[];
  language?: string;
}

export interface TranscriptionResponse {
  success: boolean;
  message: string;
  total_segments: number;
  total_duration: number;
  segments: AudioSegment[];
}

// Training Types
export interface TrainingConfig {
  model_name: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  gradient_accumulation_steps?: number;
  save_step?: number;
}

export interface TrainingRequest {
  config: TrainingConfig;
  dataset_path?: string;
}

export interface TrainingProgress {
  status: TrainingStatus;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  loss?: number;
  learning_rate?: number;
  time_elapsed: number;
  time_remaining?: number;
  gpu_utilization?: number;
  gpu_memory_used?: number;
  gpu_memory_total?: number;
  message?: string;
}

export interface TrainingResponse {
  success: boolean;
  message: string;
  model_id?: string;
  training_started: boolean;
}

// Inference Types
export interface GenerateRequest {
  model_id: string;
  text: string;
  language?: string;
  temperature?: number;
  speed?: number;
}

export interface GenerateResponse {
  success: boolean;
  message: string;
  audio_path?: string;
  audio_url?: string;
  duration?: number;
}

// Model Management Types
export interface ModelMetadata {
  model_id: string;
  model_name: string;
  created_at: string;
  training_duration: number;
  total_audio_duration: number;
  num_clips: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  final_loss?: number;
  file_size: number;
  quality_score?: number;
}

export interface ModelInfo {
  metadata: ModelMetadata;
  training_logs?: string;
}

export interface ModelsListResponse {
  success: boolean;
  models: ModelMetadata[];
  total_count: number;
}

export interface ModelUpdateRequest {
  model_name?: string;
}

export interface ModelDeleteResponse {
  success: boolean;
  message: string;
}

// System Types
export interface GPUInfo {
  available: boolean;
  device_name?: string;
  cuda_version?: string;
  device_count: number;
  memory_total?: number;
  memory_allocated?: number;
  memory_reserved?: number;
}

export interface DiskInfo {
  total: number;
  used: number;
  free: number;
  percent_used: number;
}

export interface SystemInfo {
  cuda_available: boolean;
  gpu_info?: GPUInfo;
  disk_info: DiskInfo;
  python_version: string;
  torch_version: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  cuda_available: boolean;
}

// UI State Types
export interface UploadedFile extends AudioFileInfo {
  file_id: string;
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'uploaded' | 'error';
  error?: string;
}

export interface ProcessingStatus {
  status: 'idle' | 'processing' | 'completed' | 'error';
  message?: string;
  progress?: number;
}

// Validation Types
export interface ValidationResult {
  valid: boolean;
  num_segments: number;
  total_duration: number;
  total_duration_minutes: number;
  average_segment_duration: number;
  warnings: string[];
  errors: string[];
}
