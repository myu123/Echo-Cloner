"""
Inference service for text-to-speech generation using trained XTTS models

Note: torchaudio.load is patched in app.main to use soundfile as fallback
when torchcodec is not installed (torchaudio >=2.9).
"""
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List
import time
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.tts.models.xtts import Xtts
from ..models.config import settings
from ..utils.cuda_utils import get_optimal_device
from ..utils.file_utils import generate_unique_id


class InferenceService:
    """Handles text-to-speech generation using trained XTTS models"""

    def __init__(self):
        self.device = get_optimal_device()
        self.current_model = None
        self.current_model_id = None
        self.config = None

    def load_model(self, model_dir: Path) -> bool:
        """
        Load a trained XTTS model
        Returns True if successful
        """
        try:
            print(f"Loading model from {model_dir}...")

            # Load config
            config_path = model_dir / "config.json"
            if config_path.exists():
                self.config = XttsConfig()
                self.config.load_json(str(config_path))
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")

            # Load model checkpoint (best_model.pth, model.pth, or latest checkpoint_*.pth)
            checkpoint_path = model_dir / "best_model.pth"
            if not checkpoint_path.exists():
                checkpoint_path = model_dir / "model.pth"
            if not checkpoint_path.exists():
                # Look for checkpoint_*.pth (Coqui Trainer format)
                checkpoints = sorted(model_dir.glob("checkpoint_*.pth"))
                if checkpoints:
                    checkpoint_path = checkpoints[-1]
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

            # Initialize model and load weights
            self.current_model = Xtts.init_from_config(self.config)

            # Allow XTTS config classes for torch.load with weights_only safety in torch >= 2.6
            try:
                import torch.serialization
                from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
                torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, BaseAudioConfig])
            except Exception:
                pass

            self.current_model.load_checkpoint(
                self.config,
                checkpoint_path=str(checkpoint_path),
                vocab_path=str(model_dir / "vocab.json") if (model_dir / "vocab.json").exists() else None,
                use_deepspeed=False,
            )

            # Move to device
            self.current_model.to(self.device)
            self.current_model.eval()

            self.current_model_id = model_dir.name
            print(f"Model loaded successfully: {self.current_model_id}")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.current_model = None
            self.current_model_id = None
            return False

    def generate_speech(self,
                       text: str,
                       model_dir: Path,
                       output_dir: Path,
                       language: str = "en",
                       temperature: float = 0.65,
                       speed: float = 1.0,
                       repetition_penalty: float = 10.0,
                       length_penalty: float = 1.0,
                       top_k: int = 50,
                       top_p: float = 0.85) -> Optional[Path]:
        """
        Generate speech from text using a trained model
        Returns path to generated audio file
        """
        try:
            # Load model if not already loaded or if different model
            if self.current_model is None or self.current_model_id != model_dir.name:
                if not self.load_model(model_dir):
                    raise RuntimeError("Failed to load model")

            print(f"Generating speech for text: '{text[:50]}...'")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            output_filename = f"generated_{generate_unique_id()}.wav"
            output_path = output_dir / output_filename

            # Get reference audio for speaker embedding — use ALL available clips
            reference_paths = self._get_reference_audios(model_dir)
            if not reference_paths:
                raise RuntimeError("No reference audio found for speaker embedding")

            print(f"Using {len(reference_paths)} reference audio clip(s) for speaker embedding")

            with torch.no_grad():
                outputs = self.current_model.synthesize(
                    text=text,
                    config=self.config,
                    speaker_wav=reference_paths,
                    language=language,
                    temperature=temperature,
                    speed=speed,
                    enable_text_splitting=True,
                    # Speaker conditioning parameters for better voice similarity
                    gpt_cond_len=30,          # use up to 30s of reference for conditioning
                    gpt_cond_chunk_len=4,     # process in 4s chunks for finer embeddings
                    max_ref_len=60,           # allow up to 60s total reference audio
                    sound_norm_refs=True,     # normalize reference audio for consistent embeddings
                    # Generation quality parameters
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    top_k=top_k,
                    top_p=top_p,
                )

            audio = outputs["wav"]

            # Convert to numpy array for saving
            if isinstance(audio, torch.Tensor):
                audio_numpy = audio.cpu().numpy()
            elif isinstance(audio, np.ndarray):
                audio_numpy = audio
            else:
                audio_numpy = np.array(audio)

            # Ensure 1D
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.squeeze()

            # XTTS outputs at 24kHz
            output_sr = getattr(self.config, "output_sample_rate", None) or 24000
            sf.write(str(output_path), audio_numpy, output_sr)

            print(f"Speech generated successfully: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error generating speech: {e}")
            raise Exception(f"Speech generation failed: {e}")

    def _get_reference_audios(self, model_dir: Path) -> List[str]:
        """
        Get reference audio clips for speaker embedding.
        Returns list of string paths (XTTS synthesize expects str, not Path).
        Uses all available references (up to 15) for best voice similarity.
        """
        refs_dir = model_dir / "references"
        if refs_dir.exists():
            refs = sorted(refs_dir.glob("*.wav"))
            if refs:
                return [str(r) for r in refs[:15]]

        # Fallback: look for any wav in model dir
        refs = sorted(model_dir.glob("*.wav"))
        return [str(r) for r in refs[:15]]

    def validate_text(self, text: str) -> tuple[bool, str]:
        """
        Validate input text for generation
        Returns (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Text cannot be empty"

        if len(text) > 1000:
            return False, "Text is too long (max 1000 characters)"

        if len(text) < 1:
            return False, "Text is too short"

        # Check for only special characters
        if not any(c.isalnum() for c in text):
            return False, "Text must contain alphanumeric characters"

        return True, "Valid"

    def get_estimated_duration(self, text: str, speed: float = 1.0) -> float:
        """
        Estimate audio duration based on text length
        Returns duration in seconds
        """
        # Rough estimate: ~150 words per minute = 2.5 words per second
        words = len(text.split())
        base_duration = words / 2.5
        adjusted_duration = base_duration / speed
        return adjusted_duration

    def clear_model(self):
        """Clear loaded model from memory"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_id = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Model cleared from memory")
