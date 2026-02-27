/**
 * Inference page for speech generation
 */
import { useState, useEffect } from 'react';
import { AudioPlayer } from '@/components';
import { modelsAPI, generationAPI } from '@/services/api';
import type { ModelMetadata } from '@/types';
import toast from 'react-hot-toast';
import { Wand2, Loader2, Trash2 } from 'lucide-react';

const EXAMPLE_PROMPTS = [
  'Hello! This is a test of my cloned voice.',
  'The quick brown fox jumps over the lazy dog.',
  'Welcome to the future of voice synthesis technology.',
  'I hope you have a wonderful day!',
];

export const InferencePage: React.FC = () => {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [text, setText] = useState('');
  const [temperature, setTemperature] = useState(0.65);
  const [speed, setSpeed] = useState(1.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(10.0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUnloading, setIsUnloading] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState('');
  const [audioFilename, setAudioFilename] = useState('');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const result = await modelsAPI.listModels();
      setModels(result.models);
      if (result.models.length > 0 && !selectedModelId) {
        setSelectedModelId(result.models[0].model_id);
      }
    } catch (error) {
      toast.error('Failed to load models');
    }
  };

  const handleUnload = async () => {
    setIsUnloading(true);
    try {
      await generationAPI.unloadModel();
      toast.success('Model unloaded and CUDA cache cleared');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to unload model');
    } finally {
      setIsUnloading(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedModelId) {
      toast.error('Please select a model');
      return;
    }

    if (!text.trim()) {
      toast.error('Please enter some text');
      return;
    }

    if (text.length > 1000) {
      toast.error('Text is too long (max 1000 characters)');
      return;
    }

    setIsGenerating(true);
    setGeneratedAudioUrl('');

    try {
      const result = await generationAPI.generateSpeech({
        model_id: selectedModelId,
        text: text.trim(),
        temperature,
        speed,
        repetition_penalty: repetitionPenalty,
      });

      if (result.audio_url) {
        const fullUrl = result.audio_url.startsWith('http')
          ? result.audio_url
          : `http://localhost:8000${result.audio_url}`;

        setGeneratedAudioUrl(fullUrl);
        setAudioFilename(result.audio_url.split('/').pop() || 'generated.wav');
        toast.success('Speech generated successfully!');
      }
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const selectedModel = models.find((m) => m.model_id === selectedModelId);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-dark-text mb-2">Generate Speech</h1>
        <p className="text-dark-muted">
          Use your trained models to generate speech from text
        </p>
      </div>

      {/* Model Selection */}
      <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-dark-text mb-2">
            Select Model
          </label>
          {models.length === 0 ? (
            <div className="text-center py-8 text-dark-muted">
              <p>No trained models found.</p>
              <p className="text-sm mt-2">Please train a model first.</p>
            </div>
          ) : (
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-4 py-3 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_name} - {model.num_clips} clips
                </option>
              ))}
            </select>
          )}
        </div>

        {selectedModel && (
          <div className="bg-dark-bg rounded p-4 text-sm space-y-1">
            <p className="text-dark-muted">Model Info:</p>
            <p className="text-dark-text">Training Clips: {selectedModel.num_clips}</p>
            <p className="text-dark-text">Epochs: {selectedModel.epochs}</p>
            {selectedModel.final_loss && (
              <p className="text-dark-text">Final Loss: {selectedModel.final_loss.toFixed(4)}</p>
            )}
          </div>
        )}

        <div className="flex flex-col md:flex-row gap-2">
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim() || models.length === 0}
            className="w-full px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-dark-border disabled:text-dark-muted text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Wand2 className="w-5 h-5" />
                Generate Speech
              </>
            )}
          </button>

          <button
            onClick={handleUnload}
            disabled={isUnloading}
            className="w-full md:w-48 px-6 py-3 bg-dark-border hover:bg-red-600/20 text-dark-text hover:text-red-500 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
            title="Unload the currently loaded model and clear CUDA cache to free VRAM"
          >
            {isUnloading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Unloading...
              </>
            ) : (
              <>
                <Trash2 className="w-4 h-4" />
                Unload Model
              </>
            )}
          </button>
        </div>
      </div>

      {/* Text Input */}
      {models.length > 0 && (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-dark-text">Text to Speak</label>
              <span className="text-xs text-dark-muted">
                {text.length} / 1000 characters
              </span>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to generate speech..."
              rows={5}
              maxLength={1000}
              className="w-full px-4 py-3 bg-dark-bg border border-dark-border rounded text-dark-text focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {EXAMPLE_PROMPTS.map((prompt, i) => (
              <button
                key={i}
                onClick={() => setText(prompt)}
                className="px-3 py-1.5 bg-dark-bg hover:bg-dark-border border border-dark-border rounded text-xs text-dark-text transition-colors"
              >
                {prompt.substring(0, 30)}...
              </button>
            ))}
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-dark-muted mt-1">
                Lower = closer to reference voice, Higher = more varied
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Speed: {speed.toFixed(2)}x
              </label>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-dark-muted mt-1">
                Adjust playback speed
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text mb-2">
                Repetition Penalty: {repetitionPenalty.toFixed(1)}
              </label>
              <input
                type="range"
                min="1.0"
                max="20.0"
                step="0.5"
                value={repetitionPenalty}
                onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-dark-muted mt-1">
                Higher = less repetition artifacts
              </p>
            </div>
          </div>

        </div>
      )}

      {/* Generated Audio */}
      {generatedAudioUrl && (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-4">
          <h3 className="text-lg font-semibold text-dark-text">Generated Audio</h3>
          <AudioPlayer src={generatedAudioUrl} filename={audioFilename} />
          <button
            onClick={handleGenerate}
            className="w-full px-6 py-2 bg-dark-border hover:bg-primary-600/20 text-dark-text hover:text-primary-500 rounded-lg font-medium transition-colors"
          >
            Regenerate
          </button>
        </div>
      )}
    </div>
  );
};
