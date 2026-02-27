/**
 * Model card component for displaying model information
 */
import React, { useState } from 'react';
import { Calendar, Clock, FileAudio, HardDrive, Trash2, Edit2 } from 'lucide-react';
import { formatDate, formatDuration, formatFileSize, formatNumber } from '@/utils/format';
import type { ModelMetadata } from '@/types';

interface ModelCardProps {
  model: ModelMetadata;
  onDelete: (modelId: string) => void;
  onRename: (modelId: string, newName: string) => void;
  onSelect: (modelId: string) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onDelete,
  onRename,
  onSelect,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState(model.model_name);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleRename = () => {
    if (editedName.trim() && editedName !== model.model_name) {
      onRename(model.model_id, editedName.trim());
    }
    setIsEditing(false);
  };

  const handleDelete = () => {
    onDelete(model.model_id);
    setShowDeleteConfirm(false);
  };

  return (
    <div className="bg-dark-surface border border-dark-border rounded-lg p-5 hover:border-primary-500/50 transition-all">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        {isEditing ? (
          <input
            type="text"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
            onBlur={handleRename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleRename();
              if (e.key === 'Escape') {
                setEditedName(model.model_name);
                setIsEditing(false);
              }
            }}
            autoFocus
            className="flex-1 bg-dark-bg border border-primary-500 rounded px-2 py-1 text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        ) : (
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-dark-text">{model.model_name}</h3>
            <p className="text-xs text-dark-muted mt-1">ID: {model.model_id}</p>
          </div>
        )}

        {!isEditing && (
          <button
            onClick={() => setIsEditing(true)}
            className="p-1.5 text-dark-muted hover:text-primary-500 transition-colors"
            title="Rename"
          >
            <Edit2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2 text-sm">
          <Calendar className="w-4 h-4 text-dark-muted flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-dark-muted">Created</p>
            <p className="text-dark-text truncate">{formatDate(model.created_at)}</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-dark-muted flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-dark-muted">Training Time</p>
            <p className="text-dark-text">{formatDuration(model.training_duration)}</p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <FileAudio className="w-4 h-4 text-dark-muted flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-dark-muted">Audio Data</p>
            <p className="text-dark-text">
              {formatDuration(model.total_audio_duration)} ({model.num_clips} clips)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-sm">
          <HardDrive className="w-4 h-4 text-dark-muted flex-shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-dark-muted">Model Size</p>
            <p className="text-dark-text">{formatFileSize(model.file_size)}</p>
          </div>
        </div>
      </div>

      {/* Training Info */}
      <div className="bg-dark-bg rounded p-3 mb-4 space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-dark-muted">Epochs:</span>
          <span className="text-dark-text font-mono">{model.epochs}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-dark-muted">Batch Size:</span>
          <span className="text-dark-text font-mono">{model.batch_size}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-dark-muted">Learning Rate:</span>
          <span className="text-dark-text font-mono">{model.learning_rate.toExponential(1)}</span>
        </div>
        {model.final_loss && (
          <div className="flex justify-between">
            <span className="text-dark-muted">Final Loss:</span>
            <span className="text-dark-text font-mono">{formatNumber(model.final_loss, 4)}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      {showDeleteConfirm ? (
        <div className="bg-red-500/10 border border-red-500 rounded p-3">
          <p className="text-sm text-red-500 mb-3">Are you sure you want to delete this model?</p>
          <div className="flex gap-2">
            <button
              onClick={handleDelete}
              className="flex-1 px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors"
            >
              Delete
            </button>
            <button
              onClick={() => setShowDeleteConfirm(false)}
              className="flex-1 px-3 py-2 bg-dark-border hover:bg-dark-border/70 text-dark-text rounded text-sm font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex gap-2">
          <button
            onClick={() => onSelect(model.model_id)}
            className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded font-medium transition-colors"
          >
            Use Model
          </button>
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="px-4 py-2 bg-dark-border hover:bg-red-500/20 text-red-500 rounded transition-colors"
            title="Delete"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
