import argparse
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.utils.data
import torchaudio
import soundfile as sf

# ---------------------------------------------------------------------------
# torchaudio 2.9 requires torchcodec for torchaudio.load(), but torchcodec
# is not installed.  Monkey-patch torchaudio.load to use soundfile instead.
# This must happen BEFORE any TTS imports that call torchaudio.load().
# ---------------------------------------------------------------------------
_original_torchaudio_load = torchaudio.load


def _soundfile_torchaudio_load(
    uri, frame_offset=0, num_frames=-1, normalize=True,
    channels_first=True, format=None, buffer_size=4096, backend=None,
):
    """Drop-in replacement for torchaudio.load using soundfile."""
    try:
        return _original_torchaudio_load(
            uri, frame_offset=frame_offset, num_frames=num_frames,
            normalize=normalize, channels_first=channels_first,
            format=format, buffer_size=buffer_size, backend=backend,
        )
    except (ImportError, RuntimeError):
        pass

    # Fallback: load with soundfile
    data, sample_rate = sf.read(str(uri), dtype="float32")

    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)              # [time] -> [1, time]
    else:
        waveform = waveform.transpose(0, 1)            # [time, ch] -> [ch, time]

    if frame_offset > 0:
        waveform = waveform[:, frame_offset:]
    if num_frames > 0:
        waveform = waveform[:, :num_frames]

    if not channels_first:
        waveform = waveform.transpose(0, 1)

    return waveform, sample_rate


torchaudio.load = _soundfile_torchaudio_load

from trainer import Trainer, TrainerArgs
import trainer.generic_utils as generic_utils
import trainer.trainer as trainer_module

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
import TTS.tts.layers.xtts.trainer.gpt_trainer as gpt_trainer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig


# ---------------------------------------------------------------------------
# Monkey-patch: prevent Coqui Trainer from deleting the output folder on
# failure.  On Windows the trainer_0_log.txt is still held open by the
# logging subsystem, so the shutil.rmtree call fails with a WinError 32.
# We patch it in *both* places it can be referenced from.
# ---------------------------------------------------------------------------
def _noop_remove_experiment_folder(path):
    """No-op replacement — never delete the experiment folder."""
    pass


generic_utils.remove_experiment_folder = _noop_remove_experiment_folder
# The Trainer class also imports remove_experiment_folder at module level;
# patch it there too so that `trainer.py:1860` uses the no-op version.
if hasattr(trainer_module, "remove_experiment_folder"):
    trainer_module.remove_experiment_folder = _noop_remove_experiment_folder


# ---------------------------------------------------------------------------
# Safe dataset wrapper
# ---------------------------------------------------------------------------
def safe_dataset_patch():
    """
    Replace the XTTSDataset used by GPTTrainer.get_data_loader with a
    version that:
      - Logs the *first* error per sample so we can diagnose failures.
      - Retries with other random samples on failure.
      - Matches the exact return dict format of the base XTTSDataset.
    """
    class SafeXTTSDataset(XTTSDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._error_logged = set()

        def __getitem__(self, index):
            last_exc = None
            for attempt in range(60):
                try:
                    if self.is_eval:
                        sample = self.samples[index]
                        sample_id = str(index)
                    else:
                        langs = [k for k, v in self.samples.items() if len(v) > 0]
                        if not langs:
                            raise IndexError("No samples available in any language")
                        lang = random.choice(langs)
                        idx = random.randint(0, len(self.samples[lang]) - 1)
                        sample = self.samples[lang][idx]
                        sample_id = f"{lang}_{idx}"

                    if sample_id in self.failed_samples:
                        continue

                    tseq, audiopath, wav, cond, cond_len, cond_idxs = self.load_item(sample)

                    if (
                        wav is None
                        or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
                        or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
                    ):
                        self.failed_samples.add(sample_id)
                        continue

                    # Match the exact return format of the base XTTSDataset.__getitem__
                    return {
                        "text": tseq,
                        "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
                        "wav": wav,
                        "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
                        "filenames": audiopath,
                        "conditioning": cond.unsqueeze(1),
                        "cond_lens": torch.tensor(cond_len, dtype=torch.long)
                        if cond_len is not torch.nan
                        else torch.tensor([cond_len]),
                        "cond_idxs": torch.tensor(cond_idxs)
                        if cond_idxs is not torch.nan
                        else torch.tensor([cond_idxs]),
                    }
                except Exception as exc:
                    last_exc = exc
                    # Log the first error for each unique sample so we can diagnose
                    audio_file = sample.get("audio_file", "?") if isinstance(sample, dict) else "?"
                    err_key = f"{sample_id}_{type(exc).__name__}"
                    if err_key not in self._error_logged:
                        self._error_logged.add(err_key)
                        print(
                            f"[SafeDataset] Error loading sample {sample_id} "
                            f"({audio_file}): {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    self.failed_samples.add(sample_id)
                    continue

            # If we get here, every attempt failed
            print(
                f"[SafeDataset] FATAL: No valid sample found after 60 attempts. "
                f"Total failed samples: {len(self.failed_samples)}. "
                f"Last error: {last_exc}",
                flush=True,
            )
            raise IndexError(
                f"No valid sample found after 60 attempts. "
                f"Failed: {len(self.failed_samples)} samples. Last error: {last_exc}"
            )

    gpt_trainer.XTTSDataset = SafeXTTSDataset


# ---------------------------------------------------------------------------
# Metadata CSV formatter
# ---------------------------------------------------------------------------
def metadata_formatter(root_path: str, meta_file: str, ignored_speakers=None):
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
            wav_path = Path(root_path) / "wavs" / filename
            items.append(
                {
                    "text": text,
                    "audio_file": str(wav_path),
                    "speaker_name": speaker,
                    "root_path": str(root_path),
                    "language": "en",
                }
            )
    return items


# ---------------------------------------------------------------------------
# Model artifact discovery
# ---------------------------------------------------------------------------
def find_model_artifact(output_dir: Path) -> Optional[Path]:
    """
    Search output_dir and subdirectories for best_model.pth or model.pth.
    Coqui Trainer creates a timestamped subdirectory inside output_path.
    """
    for name in ("best_model.pth", "model.pth"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate

    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            for name in ("best_model.pth", "model.pth"):
                candidate = sub / name
                if candidate.exists():
                    return candidate

    for name in ("best_model.pth", "model.pth"):
        results = list(output_dir.rglob(name))
        if results:
            return results[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--save_step", type=int, required=True)
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()

    print(
        f"[xtts_train] Starting: run={args.run_name}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}",
        flush=True,
    )

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    base_dir = Path(args.base_dir)
    base_config = base_dir / "config.json"
    base_checkpoint = base_dir / "model.pth"
    base_vocab = base_dir / "vocab.json"
    dvae_path = base_dir / "dvae.pth"

    # --- Validate base model files ---
    missing = []
    for name, path in [
        ("config.json", base_config),
        ("model.pth", base_checkpoint),
        ("vocab.json", base_vocab),
        ("dvae.pth", dvae_path),
    ]:
        if not path.exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Missing base XTTS artifacts: {', '.join(missing)} in {base_dir}"
        )

    # --- Validate dataset ---
    metadata_csv = dataset_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_csv}")

    with open(metadata_csv, encoding="utf-8") as f:
        meta_lines = [l.strip() for l in f if l.strip()]
    print(f"[xtts_train] Dataset has {len(meta_lines)} samples", flush=True)

    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.exists():
        raise FileNotFoundError(f"wavs/ directory not found at {wavs_dir}")

    # Verify a few audio files exist and are loadable
    import torchaudio

    for line in meta_lines[:3]:
        parts = line.split("|")
        wav_name = parts[0]
        wav_path = wavs_dir / wav_name
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        try:
            audio, sr = torchaudio.load(str(wav_path))
            dur = audio.shape[-1] / sr
            print(
                f"[xtts_train]   {wav_name}: {dur:.2f}s, {sr}Hz, {audio.shape[0]}ch",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(f"Cannot load audio {wav_path}: {e}")

    # --- Apply patches ---
    safe_dataset_patch()

    try:
        import torch.serialization

        torch.serialization.add_safe_globals(
            [
                XttsConfig,
                XttsAudioConfig,
                XttsArgs,
                GPTTrainerConfig,
                GPTArgs,
                BaseDatasetConfig,
                BaseAudioConfig,
            ]
        )
    except Exception:
        pass

    # --- Build training config ---
    print("[xtts_train] Loading base config...", flush=True)
    cfg = GPTTrainerConfig()
    cfg.load_json(str(base_config))
    cfg.run_name = args.run_name
    cfg.output_path = str(output_dir)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.grad_accum_steps = args.grad_accum
    cfg.save_step = args.save_step
    cfg.eval_steps = args.save_step
    cfg.log_model_step = args.save_step
    cfg.run_eval = False
    cfg.optimizer = "AdamW"
    cfg.optimizer_params = {}

    # Force num_loader_workers=0 on Windows to avoid file-lock issues
    if sys.platform == "win32":
        cfg.num_loader_workers = 0
        cfg.num_eval_loader_workers = 0

    cfg.audio.sample_rate = 22050
    if not hasattr(cfg.audio, "dvae_sample_rate"):
        cfg.audio.dvae_sample_rate = 22050
    cfg.audio.mel_norm_file = None
    if hasattr(cfg.audio, "do_normalize"):
        cfg.audio.do_normalize = False

    cfg.datasets = [
        BaseDatasetConfig(
            formatter="inline",
            dataset_name="custom_dataset",
            path=str(dataset_dir),
            meta_file_train="metadata.csv",
            meta_file_val="metadata.csv",
            ignored_speakers=[],
            language="en",
        )
    ]

    gpt_args = GPTArgs(**cfg.model_args.to_dict())
    gpt_args.xtts_checkpoint = str(base_checkpoint)
    gpt_args.dvae_checkpoint = str(dvae_path)
    gpt_args.tokenizer_file = str(base_vocab)
    gpt_args.mel_norm_file = None
    # Enable debug logging for loading failures so we can see what's happening
    gpt_args.debug_loading_failures = True
    cfg.model_args = gpt_args

    # --- Load samples ---
    print("[xtts_train] Loading dataset samples...", flush=True)
    train_samples, eval_samples = load_tts_samples(
        cfg.datasets, eval_split=True, formatter=metadata_formatter
    )
    print(
        f"[xtts_train] Loaded {len(train_samples)} train, {len(eval_samples)} eval samples",
        flush=True,
    )

    if len(train_samples) == 0:
        raise RuntimeError("No training samples loaded — check metadata.csv and wav paths")

    # --- Quick diagnostic: try tokenizing & loading first sample ---
    print("[xtts_train] Running sample diagnostic...", flush=True)
    try:
        diag_sample = train_samples[0]
        print(f"[xtts_train]   Sample text: {diag_sample['text'][:80]}...", flush=True)
        print(f"[xtts_train]   Audio file: {diag_sample['audio_file']}", flush=True)
        print(f"[xtts_train]   Language: {diag_sample['language']}", flush=True)
        print(f"[xtts_train]   Speaker: {diag_sample.get('speaker_name', 'N/A')}", flush=True)
    except Exception as e:
        print(f"[xtts_train]   Diagnostic error: {e}", flush=True)

    # --- Initialize model ---
    print("[xtts_train] Initializing GPTTrainer model...", flush=True)
    model = GPTTrainer(cfg)
    device = next(model.parameters()).device
    print(f"[xtts_train] Model initialized on {device}", flush=True)

    # --- Try tokenizing a sample with the model's tokenizer ---
    try:
        test_text = train_samples[0]["text"]
        test_lang = train_samples[0]["language"]
        tokens = model.xtts.tokenizer.encode(test_text, test_lang)
        tokens_t = torch.IntTensor(tokens)
        has_unk = torch.any(tokens_t == 1).item()
        has_stop = torch.any(tokens_t == 0).item()
        print(
            f"[xtts_train] Tokenizer test: {len(tokens)} tokens, "
            f"UNK={has_unk}, STOP={has_stop}",
            flush=True,
        )
        if has_unk or has_stop:
            print(
                f"[xtts_train] WARNING: Tokenizer produced UNK or STOP tokens for: {test_text}",
                flush=True,
            )
    except Exception as e:
        print(f"[xtts_train] Tokenizer test FAILED: {e}", flush=True)
        traceback.print_exc()

    # --- Create trainer and run ---
    trainer_args = TrainerArgs(
        restore_path="", grad_accum_steps=args.grad_accum, use_accelerate=False
    )
    trainer_obj = Trainer(
        args=trainer_args,
        config=cfg,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"restore_step": 0, "restore_epoch": 0},
    )
    trainer_obj.keep_after_fail = True

    print("[xtts_train] Starting trainer.fit()...", flush=True)
    sys.stdout.flush()
    trainer_obj.fit()
    print("[xtts_train] trainer.fit() completed", flush=True)

    # --- Find and report model artifact ---
    model_path = find_model_artifact(output_dir)
    if model_path is None:
        print(f"[xtts_train] WARNING: No model checkpoint found in {output_dir}:", flush=True)
        for item in sorted(output_dir.rglob("*")):
            size = f"{item.stat().st_size} bytes" if item.is_file() else "dir"
            print(f"  {item} ({size})", flush=True)
        raise RuntimeError("Training finished without model checkpoint")

    print(f"ARTIFACT_MODEL={model_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
