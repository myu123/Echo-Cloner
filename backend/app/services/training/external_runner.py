import logging
import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional
import sys

from app.models.config import settings

logger = logging.getLogger(__name__)

# Module-level reference to the running subprocess so it can be killed.
_active_proc: Optional[subprocess.Popen] = None


def _find_model_artifact(output_dir: Path) -> Optional[Path]:
    """
    Search output_dir and its subdirectories for best_model.pth or model.pth.
    Coqui Trainer creates a subdirectory like '{run_name}-{timestamp}-{id}/'
    inside output_path, so we need to search recursively.
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

    # Also look for checkpoint_*.pth as fallback (saves happen on save_step)
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            checkpoints = sorted(sub.glob("checkpoint_*.pth"))
            if checkpoints:
                return checkpoints[-1]  # latest checkpoint

    for name in ("best_model.pth", "model.pth"):
        results = list(output_dir.rglob(name))
        if results:
            return results[0]

    return None


def _find_trainer_subdir(output_dir: Path) -> Optional[Path]:
    """
    Find the Coqui Trainer output subdirectory.
    The trainer creates a subdirectory like '{run_name}-{Month}-{Day}-{Year}_{time}-{id}/'.
    """
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            if (sub / "trainer_0_log.txt").exists() or (sub / "config.json").exists():
                return sub
    return None


def stop_training() -> bool:
    """Terminate the active training subprocess if one is running."""
    global _active_proc
    if _active_proc is None:
        return False

    logger.info("Stopping training subprocess (pid %s)...", _active_proc.pid)
    try:
        # On Windows, terminate() sends SIGTERM equivalent.
        # The Coqui Trainer has save_on_interrupt=True so it will try to
        # save a checkpoint before exiting.
        _active_proc.terminate()

        # Give it a few seconds to save its checkpoint gracefully.
        try:
            _active_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Subprocess did not exit after 15s, killing forcefully")
            _active_proc.kill()
            _active_proc.wait(timeout=5)

        logger.info("Training subprocess terminated")
        return True
    except Exception as e:
        logger.error("Error stopping subprocess: %s", e)
        return False


def run_external_training(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_accum: int,
    save_step: int,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Spawn xtts_subprocess_train.py as a child process and stream its output.
    Returns a dict with run metadata including the path to model artifacts.
    """
    global _active_proc

    # Resolve base model directory
    base_dir = Path(settings.BASE_DIR) / "cache" / "xtts_base"
    default_base = Path.home() / "AppData" / "Local" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
    if default_base.exists():
        base_dir = default_base
    else:
        linux_base = Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
        if linux_base.exists():
            base_dir = linux_base

    dataset_dir = settings.DATASETS_DIR / "current_training"
    run_id = uuid.uuid4().hex[:8]
    output_dir = settings.MODELS_DIR / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{run_id}"

    script_path = str((Path(__file__).parent / "xtts_subprocess_train.py").resolve())
    cmd = [
        sys.executable,
        script_path,
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--base_dir", str(base_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--grad_accum", str(grad_accum),
        "--save_step", str(save_step),
        "--run_name", run_name,
    ]

    logger.info("Starting training subprocess")
    logger.info("Command: %s", " ".join(cmd))
    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Base model dir: %s", base_dir)

    if progress_cb:
        progress_cb(f"Starting training subprocess (run {run_id})...")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(settings.BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        _active_proc = proc
    except Exception as e:
        logger.error("Failed to start training subprocess: %s", e)
        _active_proc = None
        return {
            "pid": None,
            "run_id": run_id,
            "output_dir": str(output_dir),
            "returncode": -1,
            "log_tail": [f"Failed to start subprocess: {e}"],
            "model_path": None,
            "trainer_dir": None,
        }

    log_tail: List[str] = []
    line_count = 0
    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip()
            line_count += 1
            log_tail.append(line)
            if len(log_tail) > 200:
                log_tail.pop(0)
            if any(kw in line for kw in ("EPOCH:", "TRAINING", "loss:", "FATAL", "Error", "Exception", "ARTIFACT_MODEL")):
                logger.info("[subprocess] %s", line)
            else:
                logger.debug("[subprocess] %s", line)
            if progress_cb:
                progress_cb(line)

    returncode = proc.wait()
    _active_proc = None
    logger.info("Training subprocess exited with code %d after %d output lines", returncode, line_count)

    # Find the model artifact
    model_path = _find_model_artifact(output_dir)
    trainer_dir = _find_trainer_subdir(output_dir)

    if model_path:
        logger.info("Model artifact found: %s", model_path)
    else:
        logger.warning("No model artifact (best_model.pth / model.pth) found in %s", output_dir)

    if trainer_dir:
        logger.info("Trainer output directory: %s", trainer_dir)

    return {
        "pid": proc.pid,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "returncode": returncode,
        "log_tail": log_tail,
        "model_path": str(model_path) if model_path else None,
        "trainer_dir": str(trainer_dir) if trainer_dir else None,
    }
