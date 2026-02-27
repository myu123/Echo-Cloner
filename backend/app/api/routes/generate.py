"""
Speech generation (inference) API routes
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from ...models.config import settings
from ...models.schemas import GenerateRequest, GenerateResponse
from ...services import InferenceService


router = APIRouter(prefix="/api/generate", tags=["generation"])
inference_service = InferenceService()


@router.post("", response_model=GenerateResponse)
async def generate_speech(request: GenerateRequest):
    """
    Generate speech from text using a trained model
    Returns audio file path and metadata
    """
    try:
        # Validate text
        is_valid, error_msg = inference_service.validate_text(request.text)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Find model directory
        model_dir = settings.MODELS_DIR / request.model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

        # Create output directory for generated audio
        output_dir = settings.BASE_DIR / "data" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate speech
        try:
            audio_path = inference_service.generate_speech(
                text=request.text,
                model_dir=model_dir,
                output_dir=output_dir,
                language=request.language,
                temperature=request.temperature,
                speed=request.speed,
                repetition_penalty=request.repetition_penalty,
                length_penalty=request.length_penalty,
                top_k=request.top_k,
                top_p=request.top_p,
            )

            if not audio_path or not audio_path.exists():
                raise Exception("Audio generation failed - no output file created")

            # Get audio duration
            from ...services import AudioProcessor
            audio_processor = AudioProcessor()

            try:
                duration = audio_processor.get_audio_duration(audio_path)
            except:
                duration = inference_service.get_estimated_duration(request.text, request.speed)

            # Create URL for audio file
            audio_url = f"/api/generate/audio/{audio_path.name}"

            return GenerateResponse(
                success=True,
                message="Speech generated successfully",
                audio_path=str(audio_path),
                audio_url=audio_url,
                duration=duration
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve generated audio file
    """
    try:
        output_dir = settings.BASE_DIR / "data" / "generated"
        audio_path = output_dir / filename

        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/audio/{filename}")
async def delete_audio_file(filename: str):
    """
    Delete a generated audio file
    """
    try:
        output_dir = settings.BASE_DIR / "data" / "generated"
        audio_path = output_dir / filename

        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        audio_path.unlink()

        return {"success": True, "message": "Audio file deleted"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model():
    """
    Unload currently loaded TTS model and clear CUDA cache.
    Useful to free VRAM between runs.
    """
    try:
        inference_service.clear_model()
        from ...utils.cuda_utils import clear_cuda_cache
        clear_cuda_cache()

        return {"success": True, "message": "Model unloaded and CUDA cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/estimate-duration")
async def estimate_duration(text: str, speed: float = 1.0):
    """
    Estimate audio duration for given text
    """
    try:
        duration = inference_service.get_estimated_duration(text, speed)
        words = len(text.split())

        return {
            "text_length": len(text),
            "word_count": words,
            "estimated_duration": duration,
            "speed": speed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
