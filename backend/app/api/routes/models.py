"""
Model management API routes
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List
from ...models.config import settings
from ...models.schemas import (
    ModelsListResponse,
    ModelMetadata,
    ModelInfo,
    ModelUpdateRequest,
    ModelDeleteResponse
)
from ...utils.file_utils import (
    get_all_models,
    load_model_metadata,
    save_model_metadata,
    delete_model,
    load_json
)


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
async def list_models():
    """
    Get list of all trained models with metadata
    """
    try:
        models = get_all_models(settings.MODELS_DIR)

        return ModelsListResponse(
            success=True,
            models=models,
            total_count=len(models)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """
    Get detailed information about a specific model
    """
    try:
        model_dir = settings.MODELS_DIR / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Load metadata
        metadata = load_model_metadata(model_dir)
        if not metadata:
            raise HTTPException(status_code=500, detail="Could not load model metadata")

        # Load training logs if available
        log_path = model_dir / "training_log.json"
        training_logs = None

        if log_path.exists():
            log_data = load_json(log_path)
            if log_data:
                # Convert to readable format
                training_logs = f"""
Training Configuration:
- Epochs: {log_data.get('epochs_completed', 'N/A')}
- Batch Size: {log_data['config'].get('batch_size', 'N/A')}
- Learning Rate: {log_data['config'].get('learning_rate', 'N/A')}
- Training Time: {log_data.get('training_time', 0) / 60:.2f} minutes

Loss History:
"""
                losses = log_data.get('losses', [])
                if losses:
                    # Sample losses (first, middle, last)
                    training_logs += f"- Initial Loss: {losses[0]:.4f}\n"
                    if len(losses) > 2:
                        training_logs += f"- Mid Loss: {losses[len(losses)//2]:.4f}\n"
                    training_logs += f"- Final Loss: {losses[-1]:.4f}\n"

        return ModelInfo(
            metadata=metadata,
            training_logs=training_logs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{model_id}")
async def update_model(model_id: str, update: ModelUpdateRequest):
    """
    Update model metadata (e.g., rename model)
    """
    try:
        model_dir = settings.MODELS_DIR / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Load current metadata
        metadata = load_model_metadata(model_dir)
        if not metadata:
            raise HTTPException(status_code=500, detail="Could not load model metadata")

        # Update fields
        if update.model_name:
            metadata.model_name = update.model_name

        # Save updated metadata
        if not save_model_metadata(model_dir, metadata):
            raise HTTPException(status_code=500, detail="Could not save metadata")

        return {
            "success": True,
            "message": "Model updated successfully",
            "metadata": metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
async def delete_model_endpoint(model_id: str):
    """
    Delete a trained model and all associated files
    """
    try:
        model_dir = settings.MODELS_DIR / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete model directory
        success = delete_model(model_dir)

        if not success:
            raise HTTPException(status_code=500, detail="Could not delete model")

        return ModelDeleteResponse(
            success=True,
            message=f"Model {model_id} deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/logs")
async def get_model_logs(model_id: str):
    """
    Get raw training logs for a model
    """
    try:
        model_dir = settings.MODELS_DIR / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        log_path = model_dir / "training_log.json"

        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Training logs not found")

        log_data = load_json(log_path)

        return {
            "success": True,
            "model_id": model_id,
            "logs": log_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/test")
async def test_model(model_id: str, text: str = "This is a test."):
    """
    Quick test of a model with sample text
    """
    try:
        model_dir = settings.MODELS_DIR / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Use the generate endpoint internally
        from ...services import InferenceService

        inference_service = InferenceService()

        output_dir = settings.BASE_DIR / "data" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = inference_service.generate_speech(
            text=text,
            model_dir=model_dir,
            output_dir=output_dir
        )

        if not audio_path:
            raise Exception("Generation failed")

        return {
            "success": True,
            "message": "Test generation completed",
            "audio_url": f"/api/generate/audio/{audio_path.name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
