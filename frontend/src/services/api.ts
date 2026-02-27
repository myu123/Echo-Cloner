/**
 * API service for making HTTP requests to the backend
 */
import axios from 'axios';
import type {
  UploadResponse,
  TranscriptionRequest,
  TranscriptionResponse,
  TrainingRequest,
  TrainingResponse,
  TrainingProgress,
  GenerateRequest,
  GenerateResponse,
  ModelsListResponse,
  ModelInfo,
  ModelUpdateRequest,
  ModelDeleteResponse,
  SystemInfo,
  HealthResponse,
  GPUInfo,
  ValidationResult,
} from '@/types';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Upload API
export const uploadAPI = {
  uploadFile: async (file: File, onProgress?: (progress: number) => void): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<UploadResponse>('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  },

  listFiles: async () => {
    const response = await api.get('/api/upload/files');
    return response.data;
  },

  deleteFile: async (fileId: string) => {
    const response = await api.delete(`/api/upload/files/${fileId}`);
    return response.data;
  },
};

// Transcription API
export const transcriptionAPI = {
  transcribeFiles: async (request: TranscriptionRequest): Promise<TranscriptionResponse> => {
    const response = await api.post<TranscriptionResponse>('/api/transcribe', request);
    return response.data;
  },

  getTranscriptions: async (): Promise<TranscriptionResponse> => {
    const response = await api.get<TranscriptionResponse>('/api/transcribe');
    return response.data;
  },

  clearTranscriptions: async () => {
    const response = await api.delete('/api/transcribe');
    return response.data;
  },
};

// Training API
export const trainingAPI = {
  startTraining: async (request: TrainingRequest): Promise<TrainingResponse> => {
    const response = await api.post<TrainingResponse>('/api/train', request);
    return response.data;
  },

  stopTraining: async () => {
    const response = await api.post('/api/train/stop');
    return response.data;
  },

  getStatus: async (): Promise<TrainingProgress> => {
    const response = await api.get<TrainingProgress>('/api/train/status');
    return response.data;
  },

  validateData: async (): Promise<{ success: boolean; validation: ValidationResult }> => {
    const response = await api.post('/api/train/validate');
    return response.data;
  },

  getLogs: async (): Promise<{ success: boolean; status: string; lines: string[]; total_lines: number }> => {
    const response = await api.get('/api/train/logs');
    return response.data;
  },

  getRecommendations: async (): Promise<{
    success: boolean;
    dataset: { total_minutes: number; num_segments: number; avg_segment_duration: number };
    recommendations: {
      epochs: number;
      batch_size: number;
      learning_rate: number;
      gradient_accumulation_steps: number;
      save_step: number;
      tier: string;
      tip: string;
    };
  }> => {
    const response = await api.get('/api/train/recommendations');
    return response.data;
  },
};

// Generation API
export const generationAPI = {
  generateSpeech: async (request: GenerateRequest): Promise<GenerateResponse> => {
    const response = await api.post<GenerateResponse>('/api/generate', request);
    return response.data;
  },

  getAudioFile: (filename: string): string => {
    return `${API_BASE_URL}/api/generate/audio/${filename}`;
  },

  deleteAudioFile: async (filename: string) => {
    const response = await api.delete(`/api/generate/audio/${filename}`);
    return response.data;
  },

  unloadModel: async () => {
    const response = await api.post('/api/generate/unload');
    return response.data;
  },

  estimateDuration: async (text: string, speed: number = 1.0) => {
    const response = await api.get('/api/generate/estimate-duration', {
      params: { text, speed },
    });
    return response.data;
  },
};

// Models API
export const modelsAPI = {
  listModels: async (): Promise<ModelsListResponse> => {
    const response = await api.get<ModelsListResponse>('/api/models');
    return response.data;
  },

  getModelInfo: async (modelId: string): Promise<ModelInfo> => {
    const response = await api.get<ModelInfo>(`/api/models/${modelId}`);
    return response.data;
  },

  updateModel: async (modelId: string, update: ModelUpdateRequest) => {
    const response = await api.put(`/api/models/${modelId}`, update);
    return response.data;
  },

  deleteModel: async (modelId: string): Promise<ModelDeleteResponse> => {
    const response = await api.delete<ModelDeleteResponse>(`/api/models/${modelId}`);
    return response.data;
  },

  getModelLogs: async (modelId: string) => {
    const response = await api.get(`/api/models/${modelId}/logs`);
    return response.data;
  },

  testModel: async (modelId: string, text: string = 'This is a test.') => {
    const response = await api.post(`/api/models/${modelId}/test`, null, {
      params: { text },
    });
    return response.data;
  },
};

// System API
export const systemAPI = {
  healthCheck: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/api/system/health');
    return response.data;
  },

  getCudaInfo: async (): Promise<GPUInfo> => {
    const response = await api.get<GPUInfo>('/api/system/cuda');
    return response.data;
  },

  getSystemInfo: async (): Promise<SystemInfo> => {
    const response = await api.get<SystemInfo>('/api/system/info');
    return response.data;
  },

  getMemoryInfo: async () => {
    const response = await api.get('/api/system/memory');
    return response.data;
  },

  getStorageInfo: async () => {
    const response = await api.get('/api/system/storage');
    return response.data;
  },

  clearCache: async () => {
    const response = await api.post('/api/system/clear-cache');
    return response.data;
  },
};

// Export all APIs
export default {
  upload: uploadAPI,
  transcription: transcriptionAPI,
  training: trainingAPI,
  generation: generationAPI,
  models: modelsAPI,
  system: systemAPI,
};
