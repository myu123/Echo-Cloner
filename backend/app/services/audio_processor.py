"""
Audio processing service for segmentation and preprocessing
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from pydub import AudioSegment
from pydub.silence import split_on_silence
from ..models.config import settings


class AudioProcessor:
    """Handles audio file processing, segmentation, and normalization"""

    def __init__(self):
        self.sample_rate = settings.SAMPLE_RATE
        self.min_segment_length = settings.TARGET_SEGMENT_LENGTH_MIN
        self.max_segment_length = settings.TARGET_SEGMENT_LENGTH_MAX

    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate
        """
        try:
            # Load with librosa for consistent processing
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file {file_path}: {e}")

    def get_audio_duration(self, file_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            raise Exception(f"Error getting audio duration: {e}")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalize audio to [-1, 1] range.

        XTTS expects audio with normal speech amplitude levels.
        We peak-normalize to ~0.95 to avoid clipping while keeping
        the signal at a healthy level for mel spectrogram extraction.
        """
        audio = audio.astype(np.float32)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        return audio

    def remove_silence(self, audio: np.ndarray, sr: int,
                       top_db: int = 30, frame_length: int = 2048,
                       hop_length: int = 512) -> np.ndarray:
        """Remove silence from beginning and end of audio"""
        try:
            # Trim silence
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return trimmed
        except Exception as e:
            print(f"Warning: Could not trim silence: {e}")
            return audio

    def segment_audio_by_silence(self, file_path: Path) -> List[Tuple[float, float]]:
        """
        Segment audio based on silence detection
        Returns list of (start_time, end_time) tuples in seconds
        """
        try:
            # Load audio with pydub for silence detection
            audio = AudioSegment.from_file(str(file_path))

            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=500,  # Minimum silence length in ms
                silence_thresh=-40,    # Silence threshold in dB
                keep_silence=200,      # Keep some silence at edges
                seek_step=10           # Step size for silence detection
            )

            segments = []
            current_time = 0.0

            for chunk in chunks:
                chunk_duration = len(chunk) / 1000.0  # Convert to seconds

                # Skip very short chunks
                if chunk_duration < 1.0:
                    current_time += chunk_duration
                    continue

                # If chunk is too long, we'll split it later based on timestamps
                segments.append((current_time, current_time + chunk_duration))
                current_time += chunk_duration

            return segments

        except Exception as e:
            print(f"Error in silence-based segmentation: {e}")
            return []

    def segment_audio_by_timestamps(self, audio_duration: float,
                                    word_timestamps: List[dict]) -> List[Tuple[float, float]]:
        """
        Segment audio based on word timestamps from Whisper
        Tries to create segments of TARGET_SEGMENT_LENGTH duration
        """
        segments = []
        current_start = 0.0
        current_end = 0.0

        for i, word_info in enumerate(word_timestamps):
            word_end = word_info.get('end', word_info.get('timestamp', 0))

            # Check if adding this word would exceed max segment length
            if (word_end - current_start) > self.max_segment_length:
                # Save current segment if it meets minimum length
                if (current_end - current_start) >= self.min_segment_length:
                    segments.append((current_start, current_end))
                    current_start = current_end
                else:
                    # Segment too short, extend it
                    pass

            current_end = word_end

            # Check if we've reached a good segment length
            segment_duration = current_end - current_start
            if (segment_duration >= self.min_segment_length and
                segment_duration <= self.max_segment_length):
                # Look ahead to see if next word would exceed max length
                if i + 1 < len(word_timestamps):
                    next_end = word_timestamps[i + 1].get('end', word_timestamps[i + 1].get('timestamp', 0))
                    if (next_end - current_start) > self.max_segment_length:
                        # Save segment here
                        segments.append((current_start, current_end))
                        current_start = current_end

        # Add final segment
        if (current_end - current_start) >= self.min_segment_length:
            segments.append((current_start, current_end))

        return segments

    def extract_segment(self, audio: np.ndarray, sr: int,
                       start_time: float, end_time: float) -> np.ndarray:
        """Extract a segment from audio array"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        segment = audio[start_sample:end_sample]

        # Process segment
        segment = self.remove_silence(segment, sr)
        segment = self.normalize_audio(segment)

        return segment

    def save_audio_segment(self, segment: np.ndarray, output_path: Path, sr: int = None):
        """Save audio segment to file"""
        if sr is None:
            sr = self.sample_rate

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), segment, sr)
        except Exception as e:
            raise Exception(f"Error saving audio segment to {output_path}: {e}")

    def convert_to_wav(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio file to WAV format"""
        try:
            audio, sr = self.load_audio(input_path)
            self.save_audio_segment(audio, output_path, sr)
            return True
        except Exception as e:
            print(f"Error converting {input_path} to WAV: {e}")
            return False

    def process_audio_file(self, file_path: Path, output_dir: Path,
                          segments: List[Tuple[float, float, str]]) -> List[Path]:
        """
        Process audio file and save segments
        segments: List of (start_time, end_time, text) tuples
        Returns: List of saved segment file paths
        """
        saved_files = []

        try:
            # Load audio
            audio, sr = self.load_audio(file_path)

            # Process each segment
            for idx, (start_time, end_time, text) in enumerate(segments):
                # Extract segment
                segment_audio = self.extract_segment(audio, sr, start_time, end_time)

                # Skip if segment is too short after processing
                segment_duration = len(segment_audio) / sr
                if segment_duration < 1.0:  # Minimum 1 second
                    continue

                # Create output filename
                segment_filename = f"segment_{idx:04d}.wav"
                segment_path = output_dir / segment_filename

                # Save segment
                self.save_audio_segment(segment_audio, segment_path, sr)
                saved_files.append(segment_path)

            return saved_files

        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return []

    def validate_audio_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate audio file
        Returns: (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"

            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False, "File is empty"

            if file_size > settings.MAX_UPLOAD_SIZE:
                return False, f"File size exceeds maximum ({settings.MAX_UPLOAD_SIZE / (1024**2)}MB)"

            # Check file extension
            ext = file_path.suffix.lower().lstrip('.')
            if ext not in settings.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported: {', '.join(settings.SUPPORTED_FORMATS)}"

            # Try to load audio
            try:
                audio, sr = self.load_audio(file_path)
                if len(audio) == 0:
                    return False, "Audio file contains no data"
            except Exception as e:
                return False, f"Cannot load audio: {str(e)}"

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"
