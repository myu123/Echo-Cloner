/**
 * File upload component with drag-and-drop support
 */
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileAudio } from 'lucide-react';
import { formatFileSize, formatDuration } from '@/utils/format';
import type { UploadedFile } from '@/types';

interface FileUploadProps {
  files: UploadedFile[];
  onFilesAdded: (files: File[]) => void;
  onFileRemoved: (fileId: string) => void;
  maxFiles?: number;
  maxSize?: number;
  acceptedFormats?: string[];
}

export const FileUpload: React.FC<FileUploadProps> = ({
  files,
  onFilesAdded,
  onFileRemoved,
  maxFiles = 10,
  maxSize = 500 * 1024 * 1024, // 500MB
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (files.length + acceptedFiles.length > maxFiles) {
        alert(`Maximum ${maxFiles} files allowed`);
        return;
      }

      onFilesAdded(acceptedFiles);
    },
    [files.length, maxFiles, onFilesAdded]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'],
    },
    maxSize,
    multiple: true,
  });

  const totalDuration = files.reduce((sum, file) => sum + (file.duration || 0), 0);
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8
          transition-colors cursor-pointer
          ${
            isDragActive
              ? 'border-primary-500 bg-primary-500/10'
              : 'border-dark-border hover:border-primary-500/50'
          }
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center text-center space-y-3">
          <div
            className={`
            p-4 rounded-full
            ${isDragActive ? 'bg-primary-500/20' : 'bg-dark-border'}
          `}
          >
            <Upload className={`w-8 h-8 ${isDragActive ? 'text-primary-500' : 'text-dark-muted'}`} />
          </div>

          <div>
            <p className="text-lg font-medium text-dark-text">
              {isDragActive ? 'Drop files here' : 'Drag & drop audio files'}
            </p>
            <p className="text-sm text-dark-muted mt-1">
              or click to browse (MP3, WAV, M4A, OGG, FLAC, AAC)
            </p>
          </div>

          <p className="text-xs text-dark-muted">
            Max {maxFiles} files • {formatFileSize(maxSize)} per file
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-medium text-dark-text">
              Uploaded Files ({files.length})
            </h3>
            <div className="text-xs text-dark-muted space-x-4">
              <span>Total: {formatDuration(totalDuration)}</span>
              <span>{formatFileSize(totalSize)}</span>
            </div>
          </div>

          <div className="space-y-2">
            {files.map((file) => (
              <div
                key={file.file_id}
                className="flex items-center gap-3 p-3 bg-dark-surface border border-dark-border rounded-lg"
              >
                {/* Icon */}
                <div className="flex-shrink-0 p-2 bg-primary-500/10 rounded">
                  <FileAudio className="w-5 h-5 text-primary-500" />
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-dark-text truncate">
                    {file.filename}
                  </p>
                  <div className="flex gap-3 mt-1 text-xs text-dark-muted">
                    <span>{formatFileSize(file.size)}</span>
                    {file.duration && <span>{formatDuration(file.duration)}</span>}
                    <span className="capitalize">{file.format}</span>
                  </div>

                  {/* Upload Progress */}
                  {file.status === 'uploading' && (
                    <div className="mt-2">
                      <div className="h-1 bg-dark-border rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary-500 transition-all duration-300"
                          style={{ width: `${file.progress}%` }}
                        />
                      </div>
                      <p className="text-xs text-dark-muted mt-1">{file.progress}%</p>
                    </div>
                  )}

                  {/* Status */}
                  {file.status === 'error' && (
                    <p className="text-xs text-red-500 mt-1">{file.error}</p>
                  )}
                  {file.status === 'uploaded' && (
                    <p className="text-xs text-green-500 mt-1">Uploaded</p>
                  )}
                </div>

                {/* Remove Button */}
                <button
                  onClick={() => onFileRemoved(file.file_id)}
                  className="flex-shrink-0 p-1 text-dark-muted hover:text-red-500 transition-colors"
                  disabled={file.status === 'uploading'}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
