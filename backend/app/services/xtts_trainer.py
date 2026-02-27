"""
XTTS-v2 training service for Echo Cloner
"""
import torch
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Tuple
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, GPTArgs
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
import trainer.generic_utils as trainer_utils
from TTS.tts.layers.xtts.trainer import dataset as xtts_dataset
import TTS.tts.layers.xtts.trainer.gpt_trainer as gpt_trainer
from TTS.utils.manage import ModelManager
from ..models.config import settings
from ..models.schemas import TrainingConfig, TrainingProgress, TrainingStatus, ModelMetadata
from ..utils.cuda_utils import get_optimal_device, get_gpu_utilization
from ..utils.file_utils import save_model_metadata, generate_unique_id
import shutil
import inspect


class XTTSTrainer:
    """Handles XTTS-v2 model training"""

    def __init__(self):
        self.device = get_optimal_device()
        self.is_training = False
        self.should_stop = False
        self.current_model_id = None
        self.base_model_dir = self._resolve_base_model_dir()

    def _resolve_base_model_dir(self) -> Path:
        """
        Resolve the XTTS base model directory.
        Priority:
        1) settings.XTTS_BASE_MODEL_DIR if set
        2) Common cache locations (Windows LOCALAPPDATA, Linux/macOS XDG cache)
        """
        if settings.XTTS_BASE_MODEL_DIR:
            candidate = Path(settings.XTTS_BASE_MODEL_DIR)
            if candidate.exists():
                return candidate

        candidates = []
        # Windows LOCALAPPDATA
        local_appdata = Path.home() / "AppData" / "Local" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
        candidates.append(local_appdata)
        # Linux/macOS typical cache
        candidates.append(Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2")

        for c in candidates:
            if c.exists():
                return c

        # Fallback: ask ModelManager to download (will raise if no network)
        mgr = ModelManager()
        path, _, _ = mgr.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
        return Path(path)

    def prepare_dataset(self, processed_dir: Path, dataset_dir: Path) -> Dict:
        """
        Prepare dataset for training from processed segments
        Returns dataset info
        """
        try:
            # Wipe any previous training dataset so we start clean
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Collect all segments from processed directory
            all_segments = []
            total_duration = 0.0

            for file_dir in processed_dir.iterdir():
                if file_dir.is_dir():
                    segments_file = file_dir / "segments.json"
                    if segments_file.exists():
                        with open(segments_file, 'r', encoding='utf-8') as f:
                            segments = json.load(f)
                            all_segments.extend(segments)

            if not all_segments:
                raise ValueError("No segments found in processed directory")

            # Calculate total duration
            total_duration = sum(seg.get("duration", 0) for seg in all_segments)

            # Validate minimum requirements
            if total_duration < settings.MIN_TOTAL_AUDIO_DURATION:
                min_minutes = settings.MIN_TOTAL_AUDIO_DURATION / 60
                actual_minutes = total_duration / 60
                raise ValueError(
                    f"Insufficient audio data. Need at least {min_minutes:.1f} minutes, "
                    f"but only have {actual_minutes:.1f} minutes"
                )

            if len(all_segments) < settings.MIN_SEGMENTS_FOR_TRAINING:
                raise ValueError(
                    f"Insufficient audio segments. Need at least {settings.MIN_SEGMENTS_FOR_TRAINING}, "
                    f"but only have {len(all_segments)} segments"
                )

            # Create metadata.csv for XTTS training
            metadata_path = dataset_dir / "metadata.csv"
            wavs_dir = dataset_dir / "wavs"
            wavs_dir.mkdir(exist_ok=True)

            import soundfile as sf
            import librosa

            written_count = 0
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for idx, segment in enumerate(all_segments):
                    source_path = Path(segment["audio_path"])
                    if not source_path.exists():
                        continue

                    dest_filename = f"audio_{idx:05d}.wav"
                    dest_path = wavs_dir / dest_filename

                    # Load, peak-normalize, and re-save the audio so XTTS
                    # receives properly leveled waveforms regardless of
                    # how the original segments were processed.
                    try:
                        audio, sr = librosa.load(str(source_path), sr=settings.SAMPLE_RATE, mono=True)
                        peak = max(abs(audio.max()), abs(audio.min()))
                        if peak > 0:
                            audio = audio / peak * 0.95
                        sf.write(str(dest_path), audio, sr)
                    except Exception as e:
                        print(f"Warning: failed to normalize {source_path}, copying raw: {e}")
                        shutil.copy2(source_path, dest_path)

                    text = segment["text"].replace("|", " ").strip()
                    if text:
                        f.write(f"{dest_filename}|{text}|speaker\n")
                        written_count += 1

            print(f"Wrote {written_count} entries to metadata.csv")

            dataset_info = {
                "num_segments": len(all_segments),
                "total_duration": total_duration,
                "average_duration": total_duration / len(all_segments) if all_segments else 0,
                "metadata_path": str(metadata_path),
                "wavs_dir": str(wavs_dir)
            }

            print(f"Dataset prepared: {len(all_segments)} segments, {total_duration/60:.2f} minutes")
            return dataset_info

        except Exception as e:
            raise Exception(f"Error preparing dataset: {e}")

    def train_model(self,
                   config: TrainingConfig,
                   dataset_dir: Path,
                   output_dir: Path,
                   progress_callback: Optional[Callable[[TrainingProgress], None]] = None) -> ModelMetadata:
        """
        Train XTTS model with the given configuration
        progress_callback: Function to call with training progress updates
        """
        if self.is_training:
            raise RuntimeError("Training is already in progress")

        self.is_training = True
        self.should_stop = False
        start_time = time.time()
        model = None

        try:
            # Generate model ID
            model_id = generate_unique_id()
            self.current_model_id = model_id
            model_output_dir = output_dir / model_id
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare dataset info
            metadata_path = dataset_dir / "metadata.csv"
            wavs_dir = dataset_dir / "wavs"

            if not metadata_path.exists():
                raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

            # Count total segments
            with open(metadata_path, 'r', encoding='utf-8') as f:
                num_segments = len(f.readlines())

            # Resolve base model artifacts
            base_dir = self.base_model_dir
            base_config = base_dir / "config.json"
            base_checkpoint = base_dir / "model.pth"
            base_vocab = base_dir / "vocab.json"
            base_speakers = base_dir / "speakers_xtts.pth"

            if not base_config.exists() or not base_checkpoint.exists():
                raise FileNotFoundError(f"XTTS base model not found at {base_dir}. Please download it first.")

            # Initialize GPTTrainer config (training-capable wrapper around XTTS)
            print("Initializing XTTS configuration from base checkpoint...")
            gpt_config = GPTTrainerConfig()
            gpt_config.load_json(str(base_config))

            # Override training params
            gpt_config.run_name = config.model_name
            gpt_config.num_epochs = config.epochs
            gpt_config.batch_size = config.batch_size
            gpt_config.eval_batch_size = config.batch_size
            gpt_config.lr = config.learning_rate
            gpt_config.grad_accum_steps = config.gradient_accumulation_steps
            gpt_config.save_step = config.save_step
            gpt_config.eval_steps = config.save_step
            gpt_config.test_delay_epochs = 0
            gpt_config.log_model_step = config.save_step
            gpt_config.num_loader_workers = min(4, settings.MAX_WORKERS)
            gpt_config.run_eval = False
            gpt_config.model_dir = str(base_dir)
            gpt_config.optimizer = "AdamW"
            gpt_config.optimizer_params = {}
            gpt_config.output_path = str(model_output_dir)
            # Avoid mel norm file lookups (local or remote)
            if hasattr(gpt_config, "audio"):
                gpt_config.audio.mel_norm_file = None
                if hasattr(gpt_config.audio, "do_normalize"):
                    gpt_config.audio.do_normalize = False

            # Dataset config
            dataset_cfg = BaseDatasetConfig(
                formatter="inline",  # handled via custom formatter
                dataset_name="custom_dataset",
                path=str(dataset_dir),
                meta_file_train="metadata.csv",
                meta_file_val="metadata.csv",
                ignored_speakers=[],
                language="en",
            )
            gpt_config.datasets = [dataset_cfg]

            # Model args: use GPTArgs (extends XttsArgs)
            gpt_args = GPTArgs(**gpt_config.model_args.to_dict())
            gpt_args.xtts_checkpoint = str(base_checkpoint)
            # Disable remote/local mel norm file to avoid fs/permissions issues
            gpt_args.mel_norm_file = None
            # Set DVAE checkpoint path from base model directory
            dvae_path = base_dir / "dvae.pth"
            if dvae_path.exists():
                gpt_args.dvae_checkpoint = str(dvae_path)
            else:
                raise FileNotFoundError(
                    f"DVAE checkpoint not found at {dvae_path}. Please ensure the XTTS base model download includes dvae.pth."
                )
            if base_vocab.exists():
                gpt_args.tokenizer_file = str(base_vocab)
            gpt_config.model_args = gpt_args

            # Audio config
            gpt_config.audio.sample_rate = settings.SAMPLE_RATE
            # Provide DVAE sample rate expected by GPTTrainer
            if not hasattr(gpt_config.audio, "dvae_sample_rate"):
                gpt_config.audio.dvae_sample_rate = settings.SAMPLE_RATE

            # Load samples
            print("Loading dataset samples...")
            train_samples, eval_samples = load_tts_samples(
                gpt_config.datasets,
                eval_split=True,
                formatter=self._metadata_formatter,
            )

            # Allow config classes for torch.load with weights_only safety in torch >= 2.6
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([
                    XttsConfig, XttsAudioConfig, XttsArgs,
                    GPTTrainerConfig, GPTArgs,
                    BaseDatasetConfig, BaseAudioConfig
                ])
            except Exception:
                pass

            # Safe dataset wrapper to avoid recursion on failed samples
            class SafeXTTSDataset(XTTSDataset):
                def __getitem__(self_ds, index):
                    for _ in range(50):
                        try:
                            if self_ds.is_eval:
                                sample = self_ds.samples[index]
                                sample_id = str(index)
                            else:
                                langs = [k for k, v in self_ds.samples.items() if len(v) > 0]
                                if not langs:
                                    raise IndexError("No samples available")
                                lang = random.choice(langs)
                                idx = random.randint(0, len(self_ds.samples[lang]) - 1)
                                sample = self_ds.samples[lang][idx]
                                sample_id = f"{lang}_{idx}"
                            if sample_id in self_ds.failed_samples:
                                continue
                            tseq, audiopath, wav, cond, cond_len, cond_idxs = self_ds.load_item(sample)
                            if (
                                wav is None
                                or (self_ds.max_wav_len is not None and wav.shape[-1] > self_ds.max_wav_len)
                                or (self_ds.max_text_len is not None and tseq.shape[0] > self_ds.max_text_len)
                            ):
                                self_ds.failed_samples.add(sample_id)
                                continue
                            return {
                                "text": tseq,
                                "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
                                "wav": wav,
                                "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
                                "filenames": audiopath,
                                "conditioning": cond.unsqueeze(1),
                                "cond_lens": torch.tensor(cond_len, dtype=torch.long),
                                "cond_idxs": torch.tensor(cond_idxs, dtype=torch.long) if cond_idxs is not None else torch.nan,
                            }
                        except Exception:
                            continue
                    raise IndexError("No valid sample found after multiple attempts")

            # Apply patches so GPTTrainer uses safe dataset
            xtts_dataset.XTTSDataset = SafeXTTSDataset
            gpt_trainer.XTTSDataset = SafeXTTSDataset

            # Trainer args
            trainer_args = TrainerArgs(
                restore_path="",  # do not restore trainer state from base checkpoint
                grad_accum_steps=config.gradient_accumulation_steps,
                use_accelerate=False,
            )

            # Send initial progress
            if progress_callback:
                progress_callback(TrainingProgress(
                    status=TrainingStatus.TRAINING,
                    current_epoch=0,
                    total_epochs=config.epochs,
                    current_step=0,
                    total_steps=0,
                    message="Starting XTTS fine-tuning...",
                    gpu_memory_total=get_gpu_utilization().get("gpu_memory_total"),
                ))

            if model is None:
                raise RuntimeError("GPTTrainer initialization failed; model is None")

            trainer = Trainer(
                args=trainer_args,
                config=gpt_config,
                output_path=str(model_output_dir),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
                training_assets={"restore_step": 0, "restore_epoch": 0},
            )

            total_steps_estimate = max(1, int(len(train_samples) / max(1, config.batch_size)) * config.epochs)
            self._attach_progress_hooks(trainer, config, total_steps_estimate, progress_callback, start_time)

            # Fit (coarse progress only; Trainer handles internal logging)
            # Monkey-patch remove_experiment_folder to ignore Windows file locks
            trainer_utils.remove_experiment_folder = lambda path: None
            import trainer.generic_utils as generic_utils
            generic_utils.remove_experiment_folder = lambda path: None
            trainer.keep_after_fail = True

            trainer.fit()

            # Locate final/best checkpoint
            final_checkpoint_path = model_output_dir / "best_model.pth"
            if not final_checkpoint_path.exists():
                # Trainer may save model.pth
                alt = model_output_dir / "model.pth"
                if alt.exists():
                    final_checkpoint_path = alt

            # Save training logs (coarse)
            training_log = {
                "config": config.dict(),
                "training_time": time.time() - start_time,
                "num_segments": num_segments,
                "epochs_completed": config.epochs,
            }

            log_path = model_output_dir / "training_log.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(training_log, f, indent=2)

            # Copy reference audio
            self._save_references(wavs_dir, model_output_dir)

            # Create model metadata
            total_duration = num_segments * 5.0  # rough
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=config.model_name,
                created_at=datetime.now(),
                training_duration=time.time() - start_time,
                total_audio_duration=total_duration,
                num_clips=num_segments,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                final_loss=None,
                file_size=final_checkpoint_path.stat().st_size if final_checkpoint_path.exists() else 0,
                quality_score=None
            )

            # Save metadata
            save_model_metadata(model_output_dir, metadata)

            # Send completion progress
            if progress_callback:
                progress_callback(TrainingProgress(
                    status=TrainingStatus.COMPLETED,
                    current_epoch=config.epochs,
                    total_epochs=config.epochs,
                    current_step=0,
                    total_steps=0,
                    loss=None,
                    time_elapsed=time.time() - start_time,
                    message="Training completed successfully!"
                ))

            print(f"Training completed! Model saved to {model_output_dir}")
            return metadata

        except Exception as e:
            # Send error progress
            if progress_callback:
                progress_callback(TrainingProgress(
                    status=TrainingStatus.FAILED,
                    message=f"Training failed: {str(e)}"
                ))
            raise Exception(f"Training failed: {e}")

        finally:
            self.is_training = False
            self.current_model_id = None

    def _save_references(self, wavs_dir: Path, model_output_dir: Path, max_refs: int = 15):
        """Copy a few training clips as reference samples for inference."""
        ref_dir = model_output_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for wav in sorted(wavs_dir.glob("*.wav")):
            dest = ref_dir / f"ref_{count:03d}.wav"
            try:
                shutil.copy2(wav, dest)
                count += 1
            except Exception as e:
                print(f"Warning: could not copy reference {wav}: {e}")
            if count >= max_refs:
                break

    def stop_training(self):
        """Request to stop training"""
        if self.is_training:
            print("Stopping training...")
            self.should_stop = True
            return True
        return False

    def _attach_progress_hooks(
        self,
        trainer: Trainer,
        config: TrainingConfig,
        total_steps: int,
        progress_callback: Optional[Callable[[TrainingProgress], None]],
        start_time: float,
    ):
        """
        Monkey-patch trainer's logging to emit richer progress updates to the websocket.
        """
        if progress_callback is None:
            return

        # Try to capture the trainer's log hook
        if hasattr(trainer, "_log_train_step"):
            original_log_train_step = trainer._log_train_step
        else:
            original_log_train_step = None

        def _progress_hook(loss=None, step=None, epoch=None):
            try:
                gpu_stats = get_gpu_utilization()
                elapsed = time.time() - start_time
                remaining = None
                if step is not None and step > 0:
                    steps_per_sec = step / elapsed if elapsed > 0 else None
                    if steps_per_sec:
                        remaining = (total_steps - step) / steps_per_sec

                progress_callback(
                    TrainingProgress(
                        status=TrainingStatus.TRAINING,
                        current_epoch=(epoch or 0) + 1 if epoch is not None else 0,
                        total_epochs=config.epochs,
                        current_step=step or 0,
                        total_steps=total_steps,
                        loss=loss if loss is not None else None,
                        learning_rate=config.learning_rate,
                        time_elapsed=elapsed,
                        time_remaining=remaining,
                        gpu_utilization=gpu_stats["gpu_utilization"],
                        gpu_memory_used=gpu_stats["gpu_memory_used"],
                        gpu_memory_total=gpu_stats["gpu_memory_total"],
                        message="Training in progress",
                    )
                )
            except Exception:
                # Silent fail for progress hook; should not break training
                pass

        if original_log_train_step:
            def patched_log_train_step(*args, **kwargs):
                loss_val = kwargs.get("loss")
                step_val = kwargs.get("step") or kwargs.get("global_step")
                epoch_val = kwargs.get("epoch")

                # Try positional args if needed
                if loss_val is None and len(args) >= 2:
                    loss_val = args[1]
                if step_val is None and len(args) >= 3:
                    step_val = args[2]
                if epoch_val is None and len(args) >= 4:
                    epoch_val = args[3]

                _progress_hook(loss=loss_val, step=step_val, epoch=epoch_val)
                return original_log_train_step(*args, **kwargs)

            trainer._log_train_step = patched_log_train_step  # type: ignore[attr-defined]

        # Also add a fallback timer-based heartbeat every N seconds
        trainer._tts_progress_hook = _progress_hook  # stash for potential external use

    def _metadata_formatter(self, root_path: str, meta_file: str, ignored_speakers=None):
        """
        Simple formatter for metadata.csv lines: filename|text|speaker
        Returns list of dicts with audio_file, text, speaker_name.
        """
        items = []
        meta_path = Path(root_path) / meta_file
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue
                filename, text, speaker = parts[0], parts[1], parts[2]
                if ignored_speakers and speaker in ignored_speakers:
                    continue
                items.append(
                    {
                        "text": text,
                        "audio_file": str(Path(root_path) / "wavs" / filename),
                        "speaker_name": speaker,
                        "root_path": str(root_path),
                        "language": "en",
                    }
                )
        return items

    def validate_training_data(self, processed_dir: Path) -> Dict:
        """
        Validate that we have enough data for training
        Returns validation results
        """
        try:
            # Collect all segments
            all_segments = []

            for file_dir in processed_dir.iterdir():
                if file_dir.is_dir():
                    segments_file = file_dir / "segments.json"
                    if segments_file.exists():
                        with open(segments_file, 'r', encoding='utf-8') as f:
                            segments = json.load(f)
                            all_segments.extend(segments)

            total_duration = sum(seg.get("duration", 0) for seg in all_segments)
            num_segments = len(all_segments)

            # Check requirements
            min_duration_met = total_duration >= settings.MIN_TOTAL_AUDIO_DURATION
            recommended_duration_met = total_duration >= settings.RECOMMENDED_MIN_AUDIO_DURATION
            min_segments_met = num_segments >= settings.MIN_SEGMENTS_FOR_TRAINING

            warnings = []
            errors = []

            if not min_duration_met:
                min_minutes = settings.MIN_TOTAL_AUDIO_DURATION / 60
                actual_minutes = total_duration / 60
                errors.append(
                    f"Insufficient audio: need {min_minutes:.1f}+ minutes, have {actual_minutes:.1f} minutes"
                )

            if not recommended_duration_met and min_duration_met:
                rec_minutes = settings.RECOMMENDED_MIN_AUDIO_DURATION / 60
                actual_minutes = total_duration / 60
                warnings.append(
                    f"Recommended {rec_minutes:.1f}+ minutes for best quality, have {actual_minutes:.1f} minutes"
                )

            if not min_segments_met:
                errors.append(
                    f"Insufficient segments: need {settings.MIN_SEGMENTS_FOR_TRAINING}+, have {num_segments}"
                )

            if total_duration > settings.MAX_AUDIO_DURATION:
                max_hours = settings.MAX_AUDIO_DURATION / 3600
                warnings.append(
                    f"Diminishing returns beyond {max_hours:.1f} hours of audio"
                )

            is_valid = len(errors) == 0

            return {
                "valid": is_valid,
                "num_segments": num_segments,
                "total_duration": total_duration,
                "total_duration_minutes": total_duration / 60,
                "average_segment_duration": total_duration / num_segments if num_segments > 0 else 0,
                "warnings": warnings,
                "errors": errors
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
