/**
 * Training page for model training
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { TrainingProgress } from '@/components';
import { trainingAPI } from '@/services/api';
import { trainingWebSocket } from '@/services/websocket';
import type { TrainingConfig, TrainingProgress as TrainingProgressType, ValidationResult } from '@/types';
import toast from 'react-hot-toast';
import { Play, Square, AlertCircle, CheckCircle, Info, Terminal, ChevronDown, Lightbulb } from 'lucide-react';
import { formatDuration } from '@/utils/format';

interface TrainingRecommendations {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  gradient_accumulation_steps: number;
  save_step: number;
  tier: string;
  tip: string;
}

export const TrainingPage: React.FC = () => {
  const navigate = useNavigate();
  const [modelName, setModelName] = useState('');
  const [epochs, setEpochs] = useState(15);
  const [batchSize, setBatchSize] = useState(2);
  const [learningRate, setLearningRate] = useState('5e-6');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(true);
  const [progress, setProgress] = useState<TrainingProgressType | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [recommendations, setRecommendations] = useState<TrainingRecommendations | null>(null);
  const [recDataset, setRecDataset] = useState<{ total_minutes: number; num_segments: number } | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const pollIntervalRef = useRef<number | null>(null);

  // Scroll log panel to bottom when new lines arrive
  useEffect(() => {
    if (showLogs && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logLines, showLogs]);

  // Fetch training status via HTTP (used as fallback and on initial load)
  const fetchStatus = useCallback(async () => {
    try {
      const status = await trainingAPI.getStatus();
      if (status && status.status !== 'idle') {
        setProgress(status);
      }
    } catch {
      // Silently ignore poll errors
    }
  }, []);

  // Fetch training logs via HTTP
  const fetchLogs = useCallback(async () => {
    try {
      const result = await trainingAPI.getLogs();
      if (result.lines && result.lines.length > 0) {
        setLogLines(result.lines);
      }
    } catch {
      // Silently ignore
    }
  }, []);

  // Start HTTP polling when training is active
  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) return;
    pollIntervalRef.current = window.setInterval(() => {
      fetchStatus();
      fetchLogs();
    }, 2000);
  }, [fetchStatus, fetchLogs]);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    validateData();

    // Fetch current status on page load (catches in-progress training)
    fetchStatus();

    // Connect to WebSocket for training updates
    trainingWebSocket.connect();

    const unsubscribe = trainingWebSocket.onMessage((data) => {
      setProgress(data);
    });

    return () => {
      unsubscribe();
      trainingWebSocket.disconnect();
      stopPolling();
    };
  }, [fetchStatus, stopPolling]);

  // Start/stop polling based on training state
  useEffect(() => {
    const isActive = progress?.status === 'training' || progress?.status === 'processing';
    if (isActive) {
      startPolling();
      setShowLogs(true);
    } else if (progress?.status === 'completed' || progress?.status === 'failed') {
      // Do one final fetch, then stop
      fetchLogs();
      stopPolling();
    }
  }, [progress?.status, startPolling, stopPolling, fetchLogs]);

  const validateData = async () => {
    setIsValidating(true);
    try {
      const result = await trainingAPI.validateData();
      setValidation(result.validation);

      if (!result.validation.valid) {
        result.validation.errors.forEach((error) => toast.error(error));
      }

      // Fetch recommendations alongside validation
      try {
        const recResult = await trainingAPI.getRecommendations();
        if (recResult.success) {
          setRecommendations(recResult.recommendations);
          setRecDataset(recResult.dataset);
        }
      } catch {
        // Non-critical, silently ignore
      }
    } catch (error: any) {
      toast.error('Failed to validate training data');
    } finally {
      setIsValidating(false);
    }
  };

  const applyRecommendations = () => {
    if (!recommendations) return;
    setEpochs(recommendations.epochs);
    setBatchSize(recommendations.batch_size);
    setLearningRate(recommendations.learning_rate.toExponential(0));
    toast.success('Recommended settings applied');
  };

  const handleStartTraining = async () => {
    if (!modelName.trim()) {
      toast.error('Please enter a model name');
      return;
    }

    if (!validation?.valid) {
      toast.error('Please fix validation errors first');
      return;
    }

    const config: TrainingConfig = {
      model_name: modelName.trim(),
      epochs,
      batch_size: batchSize,
      learning_rate: parseFloat(learningRate),
    };

    try {
      setLogLines([]);
      setShowLogs(true);
      await trainingAPI.startTraining({ config });
      toast.success('Training started!');
      // Immediately start polling
      startPolling();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to start training');
    }
  };

  const handleStopTraining = async () => {
    try {
      await trainingAPI.stopTraining();
      toast.success('Training stop requested');
    } catch (error) {
      toast.error('Failed to stop training');
    }
  };

  const isTraining = progress?.status === 'training' || progress?.status === 'processing';

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-dark-text mb-2">Train Voice Model</h1>
        <p className="text-dark-muted">Configure and train your custom voice model</p>
      </div>

      {/* Guidance */}
      <div className="bg-dark-surface border border-dark-border rounded-lg p-4 flex flex-col md:flex-row gap-4">
        <div className="flex items-start gap-3 flex-1">
          <AlertCircle className="w-5 h-5 text-orange-400 mt-0.5" />
          <div className="space-y-1 text-sm text-dark-text">
            <p className="font-semibold text-dark-text">Best results need clean data</p>
            <p className="text-dark-muted">
              Upload 20-30 minutes of clear, single-speaker audio (minimum 10 minutes) and run "Process with Whisper."
              Segments should be 3-10 seconds, mono, 22.05 kHz. Reduce background noise and music.
            </p>
            <p className="text-dark-muted">
              Training uses your GPU. On 12GB VRAM, start with batch size 1-2 and learning rate ~5e-6. You can train multiple models; they appear in the Models tab.
            </p>
          </div>
        </div>
      </div>

      {/* Validation Status */}
      {isValidating ? (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-6 text-center">
          <p className="text-dark-muted">Validating training data...</p>
        </div>
      ) : validation && (
        <div
          className={`border rounded-lg p-6 ${
            validation.valid
              ? 'bg-green-500/10 border-green-500'
              : 'bg-red-500/10 border-red-500'
          }`}
        >
          <div className="flex items-start gap-3">
            {validation.valid ? (
              <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
            )}
            <div className="flex-1">
              <h3 className={`font-semibold mb-2 ${validation.valid ? 'text-green-500' : 'text-red-500'}`}>
                {validation.valid ? 'Dataset Ready' : 'Validation Errors'}
              </h3>
              <div className="space-y-1 text-sm">
                <p className="text-dark-text">
                  {validation.num_segments} segments • {formatDuration(validation.total_duration)} total • {validation.total_duration_minutes.toFixed(1)} minutes
                </p>
                <p className="text-dark-muted">
                  Target: 20-30 minutes (minimum 10) and 3-10s segments. Add more audio on Uploads & rerun Whisper if needed.
                </p>
                {validation.warnings.map((warning, i) => (
                  <p key={i} className="text-orange-500 flex items-start gap-2">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    {warning}
                  </p>
                ))}
                {validation.errors.map((error, i) => (
                  <p key={i} className="text-red-500">{error}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && !isTraining && progress?.status !== 'completed' && (
        <div className="bg-dark-surface border border-primary-500/30 rounded-lg p-5">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-primary-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-dark-text">
                  Recommended Settings
                  {recDataset && (
                    <span className="ml-2 text-xs font-normal text-dark-muted">
                      ({recDataset.total_minutes} min, {recDataset.num_segments} segments)
                    </span>
                  )}
                </h3>
                <button
                  onClick={applyRecommendations}
                  className="px-3 py-1.5 bg-primary-600 hover:bg-primary-700 text-white text-xs rounded font-medium transition-colors"
                >
                  Apply
                </button>
              </div>
              <p className="text-sm text-dark-muted">{recommendations.tip}</p>
              <div className="flex flex-wrap gap-3 text-xs">
                <span className="bg-dark-bg px-2 py-1 rounded text-dark-text">
                  Epochs: <strong>{recommendations.epochs}</strong>
                </span>
                <span className="bg-dark-bg px-2 py-1 rounded text-dark-text">
                  Batch Size: <strong>{recommendations.batch_size}</strong>
                </span>
                <span className="bg-dark-bg px-2 py-1 rounded text-dark-text">
                  LR: <strong>{recommendations.learning_rate.toExponential(0)}</strong>
                </span>
                <span className="bg-dark-bg px-2 py-1 rounded text-dark-text">
                  Grad Accum: <strong>{recommendations.gradient_accumulation_steps}</strong>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Training Config */}
      {!isTraining && progress?.status !== 'completed' && (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-4">
          <h3 className="text-lg font-semibold text-dark-text">Training Configuration</h3>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Model Name *
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g., My Voice Model"
                className="w-full px-4 py-2 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Epochs (5-30)
              </label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                min="5"
                max="30"
                className="w-full px-4 py-2 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Batch Size (1-4)
              </label>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="1">1</option>
                <option value="2">2 (recommended)</option>
                <option value="3">3</option>
                <option value="4">4</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Learning Rate
              </label>
              <input
                type="text"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                placeholder="5e-6"
                className="w-full px-4 py-2 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500 font-mono"
              />
            </div>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={!validation?.valid || !modelName.trim()}
            className="w-full px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-dark-border disabled:text-dark-muted text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
          >
            <Play className="w-5 h-5" />
            Start Training
          </button>
        </div>
      )}

      {/* Training Progress */}
      {progress && progress.status !== 'idle' && (
        <div>
          <TrainingProgress progress={progress} />
          {isTraining && (
            <button
              onClick={handleStopTraining}
              className="w-full mt-4 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
            >
              <Square className="w-5 h-5" />
              Stop Training
            </button>
          )}

          {progress.status === 'completed' && (
            <div className="mt-4 space-y-3">
              <button
                onClick={() => navigate('/models')}
                className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-colors"
              >
                View Models
              </button>
              <button
                onClick={() => {
                  setProgress(null);
                  setLogLines([]);
                  setShowLogs(false);
                }}
                className="w-full px-6 py-3 bg-dark-surface hover:bg-dark-border text-dark-text rounded-lg font-semibold transition-colors border border-dark-border"
              >
                Train Another Model
              </button>
            </div>
          )}

          {progress.status === 'failed' && (
            <button
              onClick={() => {
                setProgress(null);
                setLogLines([]);
              }}
              className="w-full mt-4 px-6 py-3 bg-dark-surface hover:bg-dark-border text-dark-text rounded-lg font-semibold transition-colors border border-dark-border"
            >
              Try Again
            </button>
          )}
        </div>
      )}

      {/* Training Logs */}
      {(logLines.length > 0 || isTraining) && (
        <div className="bg-dark-surface border border-dark-border rounded-lg overflow-hidden">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="w-full px-4 py-3 flex items-center justify-between text-sm font-medium text-dark-text hover:bg-dark-border/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-dark-muted" />
              <span>Training Logs ({logLines.length} lines)</span>
            </div>
            <ChevronDown className={`w-4 h-4 text-dark-muted transition-transform ${showLogs ? 'rotate-180' : ''}`} />
          </button>

          {showLogs && (
            <div className="border-t border-dark-border bg-gray-950 p-4 max-h-80 overflow-y-auto font-mono text-xs">
              {logLines.length === 0 ? (
                <p className="text-dark-muted">Waiting for training output...</p>
              ) : (
                logLines.map((line, i) => (
                  <div
                    key={i}
                    className={`py-0.5 ${
                      line.includes('ERROR') || line.includes('FATAL')
                        ? 'text-red-400'
                        : line.includes('EPOCH:') || line.includes('TRAINING')
                        ? 'text-green-400'
                        : line.includes('WARNING')
                        ? 'text-yellow-400'
                        : 'text-gray-400'
                    }`}
                  >
                    {line}
                  </div>
                ))
              )}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
