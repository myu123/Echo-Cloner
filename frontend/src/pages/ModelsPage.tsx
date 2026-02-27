/**
 * Models management page
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ModelCard } from '@/components';
import { modelsAPI } from '@/services/api';
import type { ModelMetadata } from '@/types';
import toast from 'react-hot-toast';
import { FolderOpen, RefreshCw } from 'lucide-react';

export const ModelsPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'duration'>('date');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setIsLoading(true);
    try {
      const result = await modelsAPI.listModels();
      setModels(result.models);
    } catch (error) {
      toast.error('Failed to load models');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (modelId: string) => {
    try {
      await modelsAPI.deleteModel(modelId);
      setModels((prev) => prev.filter((m) => m.model_id !== modelId));
      toast.success('Model deleted');
    } catch (error) {
      toast.error('Failed to delete model');
    }
  };

  const handleRename = async (modelId: string, newName: string) => {
    try {
      await modelsAPI.updateModel(modelId, { model_name: newName });
      setModels((prev) =>
        prev.map((m) =>
          m.model_id === modelId ? { ...m, model_name: newName } : m
        )
      );
      toast.success('Model renamed');
    } catch (error) {
      toast.error('Failed to rename model');
    }
  };

  const handleSelect = (modelId: string) => {
    navigate(`/generate?model=${modelId}`);
  };

  const sortedModels = [...models].sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.model_name.localeCompare(b.model_name);
      case 'duration':
        return b.total_audio_duration - a.total_audio_duration;
      case 'date':
      default:
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    }
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-dark-text mb-2">Voice Models</h1>
          <p className="text-dark-muted">
            Manage your trained voice models ({models.length} total)
          </p>
        </div>

        <button
          onClick={loadModels}
          disabled={isLoading}
          className="px-4 py-2 bg-dark-surface border border-dark-border hover:bg-dark-border text-dark-text rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Sort Controls */}
      {models.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-dark-muted">Sort by:</span>
          <div className="flex gap-2">
            {[
              { value: 'date', label: 'Date' },
              { value: 'name', label: 'Name' },
              { value: 'duration', label: 'Duration' },
            ].map((option) => (
              <button
                key={option.value}
                onClick={() => setSortBy(option.value as any)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  sortBy === option.value
                    ? 'bg-primary-600 text-white'
                    : 'bg-dark-surface text-dark-text hover:bg-dark-border'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Models Grid */}
      {isLoading ? (
        <div className="text-center py-12">
          <RefreshCw className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
          <p className="text-dark-muted">Loading models...</p>
        </div>
      ) : models.length === 0 ? (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-12 text-center">
          <FolderOpen className="w-16 h-16 text-dark-muted mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-dark-text mb-2">No models yet</h3>
          <p className="text-dark-muted mb-6">
            Train your first voice model to get started
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors"
          >
            Upload Audio Files
          </button>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sortedModels.map((model) => (
            <ModelCard
              key={model.model_id}
              model={model}
              onDelete={handleDelete}
              onRename={handleRename}
              onSelect={handleSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
};
