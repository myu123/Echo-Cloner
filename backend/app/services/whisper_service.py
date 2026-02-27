"""
Whisper service for transcription and automatic audio segmentation
"""
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from ..models.config import settings
from ..models.schemas import AudioSegment
from ..utils.cuda_utils import get_optimal_device
from .audio_processor import AudioProcessor

# Prefer stable-ts (better timestamps + VAD) if installed
try:
    import stable_whisper  # type: ignore

    HAS_STABLE_TS = True
except Exception:
    stable_whisper = None
    HAS_STABLE_TS = False

# Fallback to vanilla Whisper
import whisper


class WhisperService:
    """Handles Whisper transcription and automatic audio segmentation"""

    def __init__(self):
        self.model = None
        self.device = get_optimal_device()
        self.model_name = settings.WHISPER_MODEL
        self.audio_processor = AudioProcessor()
        self.use_stable_ts = bool(settings.USE_STABLE_TS and HAS_STABLE_TS)

    def load_model(self):
        """Load Whisper model (lazy loading)"""
        if self.model is None:
            impl = "stable-ts" if self.use_stable_ts else "whisper"
            print(f"Loading Whisper model ({impl}): {self.model_name} on {self.device}...")
            try:
                if self.use_stable_ts:
                    self.model = stable_whisper.load_model(
                        self.model_name,
                        device=self.device,
                        download_root=str(settings.CACHE_DIR)
                    )
                else:
                    self.model = whisper.load_model(
                        self.model_name,
                        device=self.device,
                        download_root=str(settings.CACHE_DIR)
                    )
                print("Whisper model loaded successfully")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                raise

    def transcribe_audio(self, audio_path: Path, language: str = "en") -> Dict:
        """
        Transcribe audio file using Whisper
        Returns full transcription result with word-level timestamps
        """
        self.load_model()

        try:
            print(f"Transcribing {audio_path.name}...")

            # Transcribe with word timestamps
            if self.use_stable_ts:
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    task="transcribe",
                    word_timestamps=True,
                    vad=settings.STABLE_TS_VAD,
                    regroup=False,
                    verbose=False,
                )
                # stable-ts returns an object; convert to dict for downstream compatibility
                result = result.to_dict()
            else:
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    task="transcribe",
                    word_timestamps=True,
                    verbose=False
                )

            return result

        except Exception as e:
            raise Exception(f"Error transcribing {audio_path}: {e}")

    def create_segments_from_transcription(self,
                                          transcription_result: Dict,
                                          audio_path: Path,
                                          output_dir: Path,
                                          file_id: str) -> List[AudioSegment]:
        """
        Create audio segments from Whisper transcription results
        Automatically segments based on word timestamps
        """
        segments_data = []

        try:
            # Get segments from Whisper result
            whisper_segments_raw = transcription_result.get("segments", [])

            if not whisper_segments_raw:
                print("No segments found in transcription")
                return []

            # Normalize segments to plain dicts for consistent processing
            whisper_segments = [self._segment_to_dict(seg) for seg in whisper_segments_raw]

            # Load audio for processing
            audio, sr = self.audio_processor.load_audio(audio_path)

            # Process each Whisper segment
            for seg_idx, segment in enumerate(whisper_segments):
                # Get segment timing and text
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()

                # Skip empty segments
                if not text or end_time <= start_time:
                    continue

                segment_duration = end_time - start_time

                # If segment is within our target length, use it as-is
                if (settings.TARGET_SEGMENT_LENGTH_MIN <= segment_duration <=
                    settings.TARGET_SEGMENT_LENGTH_MAX):

                    segments_data.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "duration": segment_duration
                    })

                # If segment is too long, split it using word timestamps
                elif segment_duration > settings.TARGET_SEGMENT_LENGTH_MAX:
                    words = segment.get("words", [])

                    if words:
                        # Split based on word timestamps
                        sub_segments = self._split_long_segment(words)
                        segments_data.extend(sub_segments)
                    else:
                        # No word timestamps, split evenly
                        sub_segments = self._split_evenly(start_time, end_time, text)
                        segments_data.extend(sub_segments)

                # If segment is too short, we'll try to merge with adjacent ones later
                else:
                    segments_data.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "duration": segment_duration
                    })

            # Merge short segments with adjacent ones
            segments_data = self._merge_short_segments(segments_data)

            # Create output directory for this file
            file_output_dir = output_dir / file_id
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # Process and save audio segments
            audio_segments = []
            for idx, seg_data in enumerate(segments_data):
                try:
                    # Extract audio segment
                    segment_audio = self.audio_processor.extract_segment(
                        audio, sr,
                        seg_data["start_time"],
                        seg_data["end_time"]
                    )

                    # Save audio file
                    segment_filename = f"segment_{idx:04d}.wav"
                    segment_path = file_output_dir / segment_filename

                    self.audio_processor.save_audio_segment(segment_audio, segment_path, sr)

                    # Create AudioSegment object
                    audio_segment = AudioSegment(
                        segment_id=f"{file_id}_seg_{idx:04d}",
                        start_time=seg_data["start_time"],
                        end_time=seg_data["end_time"],
                        duration=seg_data["duration"],
                        text=seg_data["text"],
                        audio_path=str(segment_path)
                    )
                    audio_segments.append(audio_segment)

                except Exception as e:
                    print(f"Error processing segment {idx}: {e}")
                    continue

            # Save segments metadata
            metadata_path = file_output_dir / "segments.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [seg.dict() for seg in audio_segments],
                    f,
                    indent=2,
                    default=str
                )

            print(f"Created {len(audio_segments)} segments from {audio_path.name}")
            return audio_segments

        except Exception as e:
            raise Exception(f"Error creating segments: {e}")

    def _segment_to_dict(self, segment: Any) -> Dict:
        """Convert stable-ts or whisper segment objects to plain dict"""
        if isinstance(segment, dict):
            return segment

        # For stable-ts Segment objects
        start = getattr(segment, "start", 0.0)
        end = getattr(segment, "end", 0.0)
        text = getattr(segment, "text", "") or ""
        words_obj = getattr(segment, "words", []) or []

        words = []
        for w in words_obj:
            words.append({
                "start": getattr(w, "start", None) or getattr(w, "timestamp", 0.0),
                "end": getattr(w, "end", None) or getattr(w, "timestamp", 0.0),
                "word": getattr(w, "word", "").strip(),
            })

        return {
            "start": start,
            "end": end,
            "text": text.strip(),
            "words": words,
        }

    def _split_long_segment(self, words: List[Dict]) -> List[Dict]:
        """Split a long segment into smaller ones based on word timestamps"""
        sub_segments = []
        current_words = []
        current_start = None
        current_duration = 0

        for word in words:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            word_text = word.get("word", "").strip()

            if current_start is None:
                current_start = word_start

            current_words.append(word_text)
            current_duration = word_end - current_start

            # Check if we should end current segment
            if current_duration >= settings.TARGET_SEGMENT_LENGTH_MIN:
                # End at sentence boundaries if possible
                if word_text.endswith(('.', '!', '?', ',')):
                    sub_segments.append({
                        "start_time": current_start,
                        "end_time": word_end,
                        "text": " ".join(current_words),
                        "duration": current_duration
                    })
                    current_words = []
                    current_start = None
                    current_duration = 0

                # Or if we're approaching max length
                elif current_duration >= settings.TARGET_SEGMENT_LENGTH_MAX * 0.9:
                    sub_segments.append({
                        "start_time": current_start,
                        "end_time": word_end,
                        "text": " ".join(current_words),
                        "duration": current_duration
                    })
                    current_words = []
                    current_start = None
                    current_duration = 0

        # Add remaining words as final segment
        if current_words and current_start is not None:
            sub_segments.append({
                "start_time": current_start,
                "end_time": words[-1].get("end", 0),
                "text": " ".join(current_words),
                "duration": current_duration
            })

        return sub_segments

    def _split_evenly(self, start_time: float, end_time: float, text: str) -> List[Dict]:
        """Split a segment evenly when no word timestamps are available"""
        total_duration = end_time - start_time
        num_segments = max(1, int(total_duration / settings.TARGET_SEGMENT_LENGTH_MAX))

        segment_duration = total_duration / num_segments
        segments = []

        # Split text roughly evenly
        words = text.split()
        words_per_segment = max(1, len(words) // num_segments)

        for i in range(num_segments):
            seg_start = start_time + (i * segment_duration)
            seg_end = start_time + ((i + 1) * segment_duration)

            # Get text for this segment
            word_start_idx = i * words_per_segment
            word_end_idx = (i + 1) * words_per_segment if i < num_segments - 1 else len(words)
            seg_text = " ".join(words[word_start_idx:word_end_idx])

            segments.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "text": seg_text,
                "duration": seg_end - seg_start
            })

        return segments

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments that are too short with adjacent ones"""
        if not segments:
            return []

        merged = []
        current_segment = segments[0].copy()

        for i in range(1, len(segments)):
            next_segment = segments[i]

            # If current segment is too short, try to merge with next
            if current_segment["duration"] < settings.TARGET_SEGMENT_LENGTH_MIN:
                # Merge with next segment
                current_segment["end_time"] = next_segment["end_time"]
                current_segment["text"] = f"{current_segment['text']} {next_segment['text']}"
                current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]

                # If merged segment is still within limits or is last segment, continue
                if current_segment["duration"] <= settings.TARGET_SEGMENT_LENGTH_MAX or i == len(segments) - 1:
                    if i == len(segments) - 1:
                        merged.append(current_segment)
                else:
                    # Merged segment too long, keep current and start new
                    merged.append(current_segment)
                    current_segment = next_segment.copy()
            else:
                # Current segment is good, save it and move to next
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Add final segment if not already added
        if current_segment not in merged:
            merged.append(current_segment)

        return merged

    def process_file(self, audio_path: Path, output_dir: Path,
                    file_id: str, language: str = "en") -> Tuple[List[AudioSegment], float]:
        """
        Complete processing pipeline: transcribe and segment
        Returns: (segments, total_duration)
        """
        try:
            # Get audio duration
            total_duration = self.audio_processor.get_audio_duration(audio_path)

            # Transcribe
            transcription_result = self.transcribe_audio(audio_path, language)

            # Create segments
            segments = self.create_segments_from_transcription(
                transcription_result,
                audio_path,
                output_dir,
                file_id
            )

            return segments, total_duration

        except Exception as e:
            raise Exception(f"Error processing file {audio_path}: {e}")
