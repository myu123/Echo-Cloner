/**
 * Upload page for audio files
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileUpload } from '@/components';
import { uploadAPI, transcriptionAPI } from '@/services/api';
import type { UploadedFile } from '@/types';
import toast from 'react-hot-toast';
import { Loader2, ArrowRight, Trash2 } from 'lucide-react';
import { formatDuration } from '@/utils/format';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<string>('');

  const handleFilesAdded = async (newFiles: File[]) => {
    const uploadPromises = newFiles.map(async (file) => {
      const fileId = `temp_${Date.now()}_${Math.random()}`;
      const uploadedFile: UploadedFile = {
        file_id: fileId,
        file,
        filename: file.name,
        size: file.size,
        format: file.name.split('.').pop() || 'unknown',
        progress: 0,
        status: 'uploading',
        uploaded_at: new Date().toISOString(),
      };

      setFiles((prev) => [...prev, uploadedFile]);

      try {
        const response = await uploadAPI.uploadFile(file, (progress) => {
          setFiles((prev) =>
            prev.map((f) => (f.file_id === fileId ? { ...f, progress } : f))
          );
        });

        setFiles((prev) =>
          prev.map((f) =>
            f.file_id === fileId
              ? {
                  ...f,
                  file_id: response.file_id,
                  status: 'uploaded',
                  progress: 100,
                  duration: response.file_info?.duration,
                }
              : f
          )
        );

        toast.success(`${file.name} uploaded successfully`);
      } catch (error: any) {
        setFiles((prev) =>
          prev.map((f) =>
            f.file_id === fileId
              ? {
                  ...f,
                  status: 'error',
                  error: error.response?.data?.detail || 'Upload failed',
                }
              : f
          )
        );
        toast.error(`Failed to upload ${file.name}`);
      }
    });

    await Promise.all(uploadPromises);
  };

  const handleFileRemoved = async (fileId: string) => {
    try {
      await uploadAPI.deleteFile(fileId);
      setFiles((prev) => prev.filter((f) => f.file_id !== fileId));
      toast.success('File removed');
    } catch (error) {
      toast.error('Failed to remove file');
    }
  };

  const handleClearAll = () => {
    files.forEach((file) => {
      if (file.status === 'uploaded') {
        uploadAPI.deleteFile(file.file_id).catch(() => {});
      }
    });
    setFiles([]);
    toast.success('All files cleared');
  };

  const handleProcessWithWhisper = async () => {
    const uploadedFiles = files.filter((f) => f.status === 'uploaded');

    if (uploadedFiles.length === 0) {
      toast.error('No uploaded files to process');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress('Starting transcription...');

    try {
      const fileIds = uploadedFiles.map((f) => f.file_id);
      const result = await transcriptionAPI.transcribeFiles({ file_ids: fileIds });

      setProcessingProgress(`Created ${result.total_segments} segments`);

      toast.success(
        `Processing complete! Created ${result.total_segments} training clips from ${formatDuration(result.total_duration)}`
      );

      // Navigate to training page after short delay
      setTimeout(() => {
        navigate('/train');
      }, 1500);
    } catch (error: any) {
      console.error('Processing error:', error);
      toast.error(error.response?.data?.detail || 'Processing failed');
      setProcessingProgress('');
    } finally {
      setIsProcessing(false);
    }
  };

  const uploadedFiles = files.filter((f) => f.status === 'uploaded');
  const totalDuration = uploadedFiles.reduce((sum, f) => sum + (f.duration || 0), 0);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-dark-text mb-2">Upload Audio Files</h1>
        <p className="text-dark-muted">
          Upload 15-30 minute audio files for Echo Cloner. The longer the better quality.
        </p>
      </div>

      <FileUpload
        files={files}
        onFilesAdded={handleFilesAdded}
        onFileRemoved={handleFileRemoved}
      />

      {uploadedFiles.length > 0 && (
        <div className="bg-dark-surface border border-dark-border rounded-lg p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-dark-text">
                Ready to Process: {uploadedFiles.length} files
              </h3>
              <p className="text-sm text-dark-muted mt-1">
                Total duration: {formatDuration(totalDuration)} ({(totalDuration / 60).toFixed(1)} minutes)
              </p>
              {totalDuration < 600 && (
                <p className="text-sm text-orange-500 mt-2">
                  Warning: Recommended minimum 10 minutes of audio for best results
                </p>
              )}
            </div>

            <button
              onClick={handleClearAll}
              className="px-4 py-2 bg-dark-border hover:bg-red-500/20 text-red-500 rounded transition-colors flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          </div>

          {isProcessing ? (
            <div className="text-center py-8">
              <Loader2 className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
              <p className="text-dark-text font-medium">{processingProgress}</p>
              <p className="text-sm text-dark-muted mt-2">
                This may take several minutes depending on file size...
              </p>
            </div>
          ) : (
            <button
              onClick={handleProcessWithWhisper}
              className="w-full px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
            >
              Process with Whisper
              <ArrowRight className="w-5 h-5" />
            </button>
          )}
        </div>
      )}
    </div>
  );
};
