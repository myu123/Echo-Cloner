"""
Training API routes
"""
import logging
import shutil
import time
import traceback
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from ...models.config import settings
from ...models.schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingProgress,
    TrainingStatus,
    ModelMetadata,
)
from ...services import XTTSTrainer
from ...services.training.external_runner import run_external_training
from ...utils.cuda_utils import check_cuda_available
from ...utils.file_utils import generate_unique_id, save_model_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/train", tags=["training"])
trainer = XTTSTrainer()
executor = ThreadPoolExecutor(max_workers=1)

# Store current training status and log lines
current_training_status: TrainingProgress = TrainingProgress(status=TrainingStatus.IDLE)
training_log_lines: List[str] = []
MAX_LOG_LINES = 500


def _update_status(
    status: TrainingStatus,
    message: str = "",
    **kwargs,
):
    """Thread-safe status update helper."""
    global current_training_status
    current_training_status = TrainingProgress(
        status=status,
        message=message[:300] if message else "",
        **kwargs,
    )
    logger.info("Training status -> %s: %s", status.value, message[:200])


def _append_log(line: str):
    """Append a line to the training log buffer."""
    global training_log_lines
    training_log_lines.append(line)
    if len(training_log_lines) > MAX_LOG_LINES:
        training_log_lines = training_log_lines[-MAX_LOG_LINES:]


def _parse_epoch_from_line(line: str) -> Optional[int]:
    """Try to extract current epoch number from a Coqui Trainer output line."""
    # Lines like " > EPOCH: 3/15"
    if "EPOCH:" in line:
        try:
            part = line.split("EPOCH:")[1].strip()
            current = int(part.split("/")[0])
            return current
        except (ValueError, IndexError):
            pass
    return None


def _parse_loss_from_line(line: str) -> Optional[float]:
    """Try to extract loss value from a Coqui Trainer output line."""
    # Lines like "   --> loss: 3.2451"  or "loss:3.2451"
    if "loss:" in line.lower():
        try:
            idx = line.lower().index("loss:")
            rest = line[idx + 5:].strip()
            # Take the first numeric token
            token = rest.split()[0].rstrip(",")
            return float(token)
        except (ValueError, IndexError):
            pass
    return None


def _parse_step_from_line(line: str) -> Optional[int]:
    """Try to extract step from Coqui Trainer output."""
    # Lines like "   > STEP: 42/1000"
    if "STEP:" in line:
        try:
            part = line.split("STEP:")[1].strip()
            current = int(part.split("/")[0])
            return current
        except (ValueError, IndexError):
            pass
    return None


def run_training_sync(config, dataset_dir, output_dir):
    """Synchronous training function to run in thread."""
    global current_training_status, training_log_lines

    training_log_lines = []
    start_time = time.time()
    current_epoch = 0
    current_step = 0
    current_loss = None

    try:
        # ── Stage 1: Prepare dataset ──
        _update_status(TrainingStatus.PROCESSING, "Preparing dataset...")
        _append_log("Preparing dataset from processed audio segments...")

        dataset_info = trainer.prepare_dataset(
            processed_dir=settings.PROCESSED_DIR,
            dataset_dir=dataset_dir,
        )

        num_segments = dataset_info.get("num_segments", 0)
        total_duration = dataset_info.get("total_duration", 0)
        _append_log(
            f"Dataset ready: {num_segments} segments, "
            f"{total_duration / 60:.1f} minutes of audio"
        )

        # ── Stage 2: Run training subprocess ──
        _update_status(
            TrainingStatus.TRAINING,
            "Launching training subprocess...",
            total_epochs=config.epochs,
        )

        def on_progress_line(line: str):
            nonlocal current_epoch, current_step, current_loss

            _append_log(line)

            # Parse structured info from the line
            epoch = _parse_epoch_from_line(line)
            if epoch is not None:
                current_epoch = epoch

            loss = _parse_loss_from_line(line)
            if loss is not None:
                current_loss = loss

            step = _parse_step_from_line(line)
            if step is not None:
                current_step = step

            elapsed = time.time() - start_time

            _update_status(
                TrainingStatus.TRAINING,
                message=line[:200],
                current_epoch=current_epoch,
                total_epochs=config.epochs,
                current_step=current_step,
                loss=current_loss,
                time_elapsed=elapsed,
            )

        run_info = run_external_training(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            grad_accum=config.gradient_accumulation_steps,
            save_step=config.save_step,
            progress_cb=on_progress_line,
        )

        rc = run_info.get("returncode", 1)
        model_path_str = run_info.get("model_path")
        trainer_dir_str = run_info.get("trainer_dir")
        out_dir = Path(run_info.get("output_dir", ""))
        log_tail = run_info.get("log_tail", [])

        elapsed = time.time() - start_time
        _append_log(f"Subprocess exited with code {rc}")

        has_artifact = model_path_str is not None
        logger.info(
            "Training subprocess result: rc=%d, has_artifact=%s, model_path=%s",
            rc, has_artifact, model_path_str,
        )

        # ── Stage 3: Register model if successful ──
        if rc == 0 and has_artifact:
            _update_status(
                TrainingStatus.PROCESSING,
                "Training complete, registering model...",
                current_epoch=config.epochs,
                total_epochs=config.epochs,
                time_elapsed=elapsed,
            )

            model_path = Path(model_path_str)
            trainer_dir = Path(trainer_dir_str) if trainer_dir_str else model_path.parent

            # Create a proper model directory
            model_id = generate_unique_id()
            model_dir = settings.MODELS_DIR / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Copy model checkpoint
            dest_model = model_dir / model_path.name
            shutil.copy2(model_path, dest_model)
            _append_log(f"Copied model checkpoint to {dest_model}")

            # Copy config.json if it exists
            config_src = trainer_dir / "config.json"
            if config_src.exists():
                shutil.copy2(config_src, model_dir / "config.json")

            # Copy vocab.json from base model
            base_vocab = trainer.base_model_dir / "vocab.json"
            if base_vocab.exists():
                shutil.copy2(base_vocab, model_dir / "vocab.json")

            # Copy reference audio clips for inference
            dataset_wavs = dataset_dir / "wavs"
            if dataset_wavs.exists():
                refs_dir = model_dir / "references"
                refs_dir.mkdir(exist_ok=True)
                wav_files = sorted(dataset_wavs.glob("*.wav"))[:15]
                for i, wav in enumerate(wav_files):
                    shutil.copy2(wav, refs_dir / f"ref_{i:03d}.wav")
                _append_log(f"Copied {len(wav_files)} reference audio clips")

            # Calculate model file size
            file_size = dest_model.stat().st_size if dest_model.exists() else 0

            # Save metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=config.model_name,
                created_at=datetime.now(),
                training_duration=elapsed,
                total_audio_duration=total_duration,
                num_clips=num_segments,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                final_loss=current_loss,
                file_size=file_size,
            )
            save_model_metadata(model_dir, metadata)
            _append_log(f"Model registered: {model_id} ({config.model_name})")

            _update_status(
                TrainingStatus.COMPLETED,
                f"Training completed! Model '{config.model_name}' is ready.",
                current_epoch=config.epochs,
                total_epochs=config.epochs,
                loss=current_loss,
                time_elapsed=elapsed,
            )

        elif rc == 0 and not has_artifact:
            # Subprocess exited cleanly but no checkpoint was saved
            tail_str = "\n".join(log_tail[-10:])
            _update_status(
                TrainingStatus.FAILED,
                "Training finished but no model checkpoint was saved. "
                "This usually means training did not complete enough steps. "
                "Try reducing save_step or increasing epochs.",
                time_elapsed=elapsed,
            )
            _append_log(f"No model artifact found. Last output:\n{tail_str}")

        else:
            # Non-zero return code
            tail_str = " | ".join(log_tail[-5:])
            error_msg = f"Training failed (exit code {rc}). {tail_str}"[:300]
            _update_status(
                TrainingStatus.FAILED,
                error_msg,
                time_elapsed=elapsed,
            )
            _append_log(f"Training failed with exit code {rc}")

    except Exception as e:
        logger.error("Training error: %s", e, exc_info=True)
        _append_log(f"ERROR: {e}")
        _append_log(traceback.format_exc())
        _update_status(
            TrainingStatus.FAILED,
            f"Training failed: {str(e)}"[:300],
            time_elapsed=time.time() - start_time,
        )
    finally:
        trainer.is_training = False


async def run_training_async(config, dataset_dir, output_dir):
    """Run training in background thread."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_training_sync, config, dataset_dir, output_dir)


@router.post("", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start XTTS model training.
    Runs training in background and provides updates via WebSocket.
    """
    try:
        # Check if already training
        if trainer.is_training:
            raise HTTPException(
                status_code=400,
                detail="Training already in progress",
            )

        # Check CUDA availability
        if not check_cuda_available():
            raise HTTPException(
                status_code=500,
                detail="CUDA not available. GPU is required for training.",
            )

        # Validate training data
        validation_result = trainer.validate_training_data(settings.PROCESSED_DIR)

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Training data validation failed",
                    "errors": validation_result.get("errors", []),
                    "warnings": validation_result.get("warnings", []),
                },
            )

        # Prepare paths
        dataset_dir = settings.DATASETS_DIR / "current_training"
        output_dir = settings.MODELS_DIR

        # Set initial status
        global current_training_status, training_log_lines
        trainer.is_training = True
        training_log_lines = []
        current_training_status = TrainingProgress(
            status=TrainingStatus.PROCESSING,
            message="Starting training...",
            total_epochs=request.config.epochs,
        )

        # Start training in background
        background_tasks.add_task(
            run_training_async,
            request.config,
            dataset_dir,
            output_dir,
        )

        logger.info("Training task started for model: %s", request.config.model_name)

        return TrainingResponse(
            success=True,
            message="Training started",
            training_started=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Friendly GPU OOM guidance
        msg = str(e)
        if "CUDA out of memory" in msg or "cuda runtime error" in msg.lower():
            msg = (
                "CUDA out of memory. Tips: set batch size to 1, increase gradient_accumulation_steps, "
                "close other GPU apps, or restart backend to clear VRAM."
            )
        raise HTTPException(status_code=500, detail=msg)


@router.post("/stop")
async def stop_training():
    """Stop current training by terminating the subprocess."""
    try:
        if not trainer.is_training:
            return {"success": False, "message": "No training in progress"}

        from ...services.training.external_runner import stop_training as kill_subprocess
        killed = kill_subprocess()

        if killed:
            _update_status(TrainingStatus.STOPPED, "Training stopped by user")
            trainer.is_training = False
            return {"success": True, "message": "Training stopped"}
        else:
            return {"success": False, "message": "No active subprocess to stop"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingProgress)
async def get_training_status():
    """Get current training status."""
    return current_training_status


@router.get("/logs")
async def get_training_logs():
    """
    Get recent training log lines.
    Returns the last N lines of subprocess output for debugging.
    """
    return {
        "success": True,
        "status": current_training_status.status.value,
        "lines": training_log_lines[-100:],
        "total_lines": len(training_log_lines),
    }


@router.post("/validate")
async def validate_training_data():
    """
    Validate that sufficient data exists for training.
    Returns validation results with errors and warnings.
    """
    try:
        validation_result = trainer.validate_training_data(settings.PROCESSED_DIR)

        return {
            "success": True,
            "validation": validation_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_training_recommendations():
    """
    Get recommended training parameters based on the current dataset size.
    Adjusts epochs, batch size, learning rate, and gradient accumulation
    based on total audio minutes and segment count.
    """
    try:
        validation_result = trainer.validate_training_data(settings.PROCESSED_DIR)
        total_minutes = validation_result.get("total_duration", 0) / 60
        num_segments = validation_result.get("num_segments", 0)
        avg_duration = validation_result.get("average_segment_duration", 5.0)

        # Recommendations scale with dataset size
        if total_minutes < 15:
            # Small dataset: train longer with smaller LR to avoid overfitting
            rec = {
                "epochs": 20,
                "batch_size": 1,
                "learning_rate": 2e-6,
                "gradient_accumulation_steps": 4,
                "save_step": max(100, num_segments // 2),
                "tier": "small",
                "tip": (
                    f"You have {total_minutes:.1f} min of audio. For better quality, "
                    "try to upload 20-30 minutes of clean, single-speaker audio. "
                    "Using conservative settings to avoid overfitting."
                ),
            }
        elif total_minutes < 30:
            # Medium dataset: balanced settings
            rec = {
                "epochs": 15,
                "batch_size": 2,
                "learning_rate": 5e-6,
                "gradient_accumulation_steps": 2,
                "save_step": max(200, num_segments),
                "tier": "medium",
                "tip": (
                    f"You have {total_minutes:.1f} min of audio. This is a good amount. "
                    "These balanced settings should produce quality results."
                ),
            }
        elif total_minutes < 60:
            # Large dataset: can be more aggressive
            rec = {
                "epochs": 10,
                "batch_size": 2,
                "learning_rate": 1e-5,
                "gradient_accumulation_steps": 1,
                "save_step": max(500, num_segments),
                "tier": "large",
                "tip": (
                    f"You have {total_minutes:.1f} min of audio. With this much data, "
                    "fewer epochs are needed. Higher learning rate can converge faster."
                ),
            }
        else:
            # Very large dataset
            rec = {
                "epochs": 8,
                "batch_size": 2,
                "learning_rate": 1e-5,
                "gradient_accumulation_steps": 1,
                "save_step": max(1000, num_segments),
                "tier": "very_large",
                "tip": (
                    f"You have {total_minutes:.1f} min of audio. Plenty of data. "
                    "Fewer epochs with higher LR will train efficiently."
                ),
            }

        return {
            "success": True,
            "dataset": {
                "total_minutes": round(total_minutes, 1),
                "num_segments": num_segments,
                "avg_segment_duration": round(avg_duration, 1),
            },
            "recommendations": rec,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
