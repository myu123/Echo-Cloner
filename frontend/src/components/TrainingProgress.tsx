/**
 * Training progress component with real-time updates
 */
import React from 'react';
import { Activity, Zap, Clock, HardDrive } from 'lucide-react';
import { formatDuration, formatPercentage, formatLoss, formatNumber } from '@/utils/format';
import type { TrainingProgress as TrainingProgressType, TrainingStatus } from '@/types';

interface TrainingProgressProps {
  progress: TrainingProgressType;
  className?: string;
}

const statusColors: Record<TrainingStatus, string> = {
  idle: 'text-gray-500',
  processing: 'text-blue-500',
  training: 'text-green-500',
  completed: 'text-emerald-500',
  failed: 'text-red-500',
  stopped: 'text-orange-500',
};

const statusLabels: Record<TrainingStatus, string> = {
  idle: 'Idle',
  processing: 'Processing',
  training: 'Training',
  completed: 'Completed',
  failed: 'Failed',
  stopped: 'Stopped',
};

export const TrainingProgress: React.FC<TrainingProgressProps> = ({ progress, className = '' }) => {
  const epochProgress =
    progress.total_epochs > 0 ? (progress.current_epoch / progress.total_epochs) * 100 : 0;

  const stepProgress =
    progress.total_steps > 0 ? (progress.current_step / progress.total_steps) * 100 : 0;

  const isTraining = progress.status === 'training' || progress.status === 'processing';

  return (
    <div className={`bg-dark-surface border border-dark-border rounded-lg p-6 ${className}`}>
      {/* Status Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div
            className={`
            flex items-center gap-2 px-3 py-1.5 rounded-full
            ${isTraining ? 'bg-green-500/10' : 'bg-dark-border'}
          `}
          >
            {isTraining && <Activity className="w-4 h-4 text-green-500 animate-pulse" />}
            <span className={`text-sm font-medium ${statusColors[progress.status]}`}>
              {statusLabels[progress.status]}
            </span>
          </div>

          {progress.message && (
            <span className="text-sm text-dark-muted">{progress.message}</span>
          )}
        </div>
      </div>

      {/* Progress Bars */}
      {isTraining && (
        <div className="space-y-4 mb-6">
          {/* Epoch Progress */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-dark-text">
                Epoch {progress.current_epoch} / {progress.total_epochs}
              </span>
              <span className="text-dark-muted">{formatPercentage(epochProgress)}</span>
            </div>
            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-300"
                style={{ width: `${epochProgress}%` }}
              />
            </div>
          </div>

          {/* Step Progress */}
          {progress.total_steps > 0 && (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-dark-text">
                  Step {progress.current_step} / {progress.total_steps}
                </span>
                <span className="text-dark-muted">{formatPercentage(stepProgress)}</span>
              </div>
              <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 transition-all duration-300"
                  style={{ width: `${stepProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stats Grid */}
      {isTraining && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Loss */}
          {progress.loss !== undefined && progress.loss !== null && (
            <div className="bg-dark-bg rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-4 h-4 text-primary-500" />
                <span className="text-xs text-dark-muted">Loss</span>
              </div>
              <p className="text-lg font-semibold text-dark-text">{formatLoss(progress.loss)}</p>
            </div>
          )}

          {/* Learning Rate */}
          {progress.learning_rate && (
            <div className="bg-dark-bg rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-xs text-dark-muted">Learning Rate</span>
              </div>
              <p className="text-lg font-semibold text-dark-text font-mono text-sm">
                {progress.learning_rate.toExponential(1)}
              </p>
            </div>
          )}

          {/* Time Elapsed */}
          <div className="bg-dark-bg rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="w-4 h-4 text-blue-500" />
              <span className="text-xs text-dark-muted">Elapsed</span>
            </div>
            <p className="text-lg font-semibold text-dark-text">
              {formatDuration(progress.time_elapsed)}
            </p>
          </div>

          {/* Time Remaining */}
          {progress.time_remaining && (
            <div className="bg-dark-bg rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-purple-500" />
                <span className="text-xs text-dark-muted">Remaining</span>
              </div>
              <p className="text-lg font-semibold text-dark-text">
                {formatDuration(progress.time_remaining)}
              </p>
            </div>
          )}

          {/* GPU Utilization */}
          {progress.gpu_utilization !== undefined && progress.gpu_utilization !== null && (
            <div className="bg-dark-bg rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-green-500" />
                <span className="text-xs text-dark-muted">GPU Usage</span>
              </div>
              <p className="text-lg font-semibold text-dark-text">
                {formatPercentage(progress.gpu_utilization)}
              </p>
            </div>
          )}

          {/* GPU Memory */}
          {progress.gpu_memory_used !== undefined &&
            progress.gpu_memory_used !== null &&
            progress.gpu_memory_total && (
            <div className="bg-dark-bg rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <HardDrive className="w-4 h-4 text-orange-500" />
                <span className="text-xs text-dark-muted">GPU Memory</span>
              </div>
              <p className="text-lg font-semibold text-dark-text">
                {formatNumber(progress.gpu_memory_used, 1)} /{' '}
                {formatNumber(progress.gpu_memory_total, 1)} GB
              </p>
            </div>
          )}
        </div>
      )}

      {/* Completed/Failed Message */}
      {(progress.status === 'completed' || progress.status === 'failed') && (
        <div
          className={`
          p-4 rounded-lg text-center
          ${progress.status === 'completed' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}
        `}
        >
          {progress.message || (progress.status === 'completed' ? 'Training completed!' : 'Training failed')}
        </div>
      )}
    </div>
  );
};
