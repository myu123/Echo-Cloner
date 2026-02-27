"""
Upload API routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import aiofiles
from typing import List
from ...models.config import settings
from ...models.schemas import UploadResponse, AudioFileInfo
from ...services import AudioProcessor
from ...utils.file_utils import generate_unique_id, clean_filename
from datetime import datetime


router = APIRouter(prefix="/api/upload", tags=["upload"])
audio_processor = AudioProcessor()


@router.post("", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file
    Supports: MP3, WAV, M4A, OGG, FLAC, AAC
    Max size: 500MB
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {', '.join(settings.SUPPORTED_FORMATS)}"
            )

        # Generate unique file ID and filename
        file_id = generate_unique_id()
        clean_name = clean_filename(file.filename)
        filename = f"{file_id}_{clean_name}"

        # Create upload path
        upload_path = settings.UPLOADS_DIR / filename

        # Save file
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()

            # Check file size
            if len(content) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024**2)}MB"
                )

            await f.write(content)

        # Validate audio file
        is_valid, error_msg = audio_processor.validate_audio_file(upload_path)
        if not is_valid:
            # Delete invalid file
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=error_msg)

        # Get audio duration
        try:
            duration = audio_processor.get_audio_duration(upload_path)
        except Exception as e:
            duration = None
            print(f"Could not get duration: {e}")

        # Create file info
        file_info = AudioFileInfo(
            filename=clean_name,
            size=len(content),
            duration=duration,
            format=file_ext,
            uploaded_at=datetime.now()
        )

        return UploadResponse(
            success=True,
            file_id=file_id,
            message="File uploaded successfully",
            file_info=file_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/files")
async def list_uploaded_files():
    """Get list of all uploaded files"""
    try:
        files = []
        for file_path in settings.UPLOADS_DIR.glob("*"):
            if file_path.is_file():
                # Extract file ID from filename
                file_id = file_path.stem.split('_')[0]

                # Get file info
                try:
                    duration = audio_processor.get_audio_duration(file_path)
                except:
                    duration = None

                files.append({
                    "file_id": file_id,
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "duration": duration,
                    "format": file_path.suffix.lstrip('.'),
                    "uploaded_at": datetime.fromtimestamp(file_path.stat().st_mtime)
                })

        return {"success": True, "files": files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
async def delete_uploaded_file(file_id: str):
    """Delete an uploaded file"""
    try:
        # Find file with this ID
        for file_path in settings.UPLOADS_DIR.glob(f"{file_id}_*"):
            if file_path.is_file():
                file_path.unlink()
                return {"success": True, "message": "File deleted"}

        raise HTTPException(status_code=404, detail="File not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
