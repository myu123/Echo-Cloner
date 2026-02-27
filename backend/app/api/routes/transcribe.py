"""
Transcription API routes
"""
import logging
import shutil

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import List, Dict
import asyncio
from ...models.config import settings
from ...models.schemas import TranscriptionRequest, TranscriptionResponse, AudioSegment
from ...services import WhisperService
from ...utils.file_utils import save_json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transcribe", tags=["transcription"])
whisper_service = WhisperService()

# Store transcription status
transcription_status: Dict[str, dict] = {}


def _clear_processed_dir():
    """Remove all existing processed segment directories.

    Called before a new transcription run so that re-processing the same
    (or different) uploads doesn't accumulate duplicate segments.
    """
    if not settings.PROCESSED_DIR.exists():
        return
    for item in list(settings.PROCESSED_DIR.iterdir()):
        try:
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
        except Exception as e:
            logger.warning("Could not remove %s: %s", item, e)


@router.post("", response_model=TranscriptionResponse)
async def transcribe_files(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    """
    Transcribe audio files and create training segments
    This processes all uploaded files and creates segmented audio clips
    """
    try:
        if not request.file_ids:
            raise HTTPException(status_code=400, detail="No file IDs provided")

        # Clear previous processed segments so re-processing doesn't
        # accumulate duplicates.
        _clear_processed_dir()

        all_segments = []
        total_duration = 0.0

        for file_id in request.file_ids:
            # Find the uploaded file
            file_path = None
            for f in settings.UPLOADS_DIR.glob(f"{file_id}_*"):
                if f.is_file():
                    file_path = f
                    break

            if not file_path:
                raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

            # Process file with Whisper
            try:
                segments, duration = whisper_service.process_file(
                    audio_path=file_path,
                    output_dir=settings.PROCESSED_DIR,
                    file_id=file_id,
                    language=request.language
                )

                all_segments.extend(segments)
                total_duration += duration

            except Exception as e:
                print(f"Error processing file {file_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Transcription failed for {file_id}: {str(e)}")

        # Save overall transcription metadata
        metadata = {
            "file_ids": request.file_ids,
            "total_segments": len(all_segments),
            "total_duration": total_duration,
            "language": request.language
        }

        metadata_path = settings.PROCESSED_DIR / "transcription_metadata.json"
        save_json(metadata, metadata_path)

        return TranscriptionResponse(
            success=True,
            message=f"Processed {len(request.file_ids)} files successfully",
            total_segments=len(all_segments),
            total_duration=total_duration,
            segments=all_segments
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.get("", response_model=TranscriptionResponse)
async def get_transcription_results():
    """
    Get current transcription results
    Returns all processed segments
    """
    try:
        all_segments = []
        total_duration = 0.0

        # Load all segment files from processed directory
        for file_dir in settings.PROCESSED_DIR.iterdir():
            if file_dir.is_dir():
                segments_file = file_dir / "segments.json"
                if segments_file.exists():
                    import json
                    with open(segments_file, 'r', encoding='utf-8') as f:
                        segments_data = json.load(f)

                    for seg_data in segments_data:
                        segment = AudioSegment(**seg_data)
                        all_segments.append(segment)
                        total_duration += segment.duration

        return TranscriptionResponse(
            success=True,
            message=f"Found {len(all_segments)} segments",
            total_segments=len(all_segments),
            total_duration=total_duration,
            segments=all_segments
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
async def clear_transcriptions():
    """
    Clear all transcription data
    Deletes all processed segments
    """
    try:
        import shutil

        # Delete processed directory contents
        if settings.PROCESSED_DIR.exists():
            for item in settings.PROCESSED_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()

        return {"success": True, "message": "All transcriptions cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{file_id}")
async def get_transcription_status(file_id: str):
    """Get transcription status for a specific file"""
    status = transcription_status.get(file_id, {"status": "not_found"})
    return status
