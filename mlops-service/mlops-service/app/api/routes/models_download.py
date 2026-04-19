from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.dependencies import get_settings

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/download/{pipeline}/model")
async def download_model(
    pipeline: str,
) -> FileResponse:
    """Download the current production model for a pipeline.
    
    Args:
        pipeline: Either 'clinical' or 'imaging'
        
    Returns:
        Model file (pkl for clinical, pth for imaging)
    """
    settings = get_settings()
    
    if pipeline == "clinical":
        model_path = settings.artifacts_root / "model" / "clinical" / "model.pkl"
        media_type = "application/octet-stream"
    elif pipeline == "imaging":
        model_path = settings.artifacts_root / "model" / "imaging" / "model.pth"
        media_type = "application/octet-stream"
    else:
        raise HTTPException(status_code=400, detail="pipeline must be 'clinical' or 'imaging'")
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found for pipeline: {pipeline}")
    
    return FileResponse(
        path=model_path,
        media_type=media_type,
        filename=model_path.name,
    )


@router.get("/download/{pipeline}/metadata")
async def download_model_metadata(
    pipeline: str,
) -> dict:
    """Download model metadata (feature names, metrics, etc.) for a pipeline."""
    settings = get_settings()
    
    if pipeline not in ["clinical", "imaging"]:
        raise HTTPException(status_code=400, detail="pipeline must be 'clinical' or 'imaging'")
    
    metadata_dir = settings.artifacts_root / "model" / pipeline
    if not metadata_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model metadata not found for: {pipeline}")
    
    result = {"pipeline": pipeline, "files": []}
    
    for f in metadata_dir.glob("*"):
        if f.is_file() and f.suffix in [".json", ".pkl", ".pth"]:
            result["files"].append({
                "name": f.name,
                "size": f.stat().st_size,
            })
    
    metrics_path = metadata_dir / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            result["metrics"] = json.load(f)
    
    importance_path = metadata_dir / "feature_importance.json"
    if importance_path.exists():
        import json
        with open(importance_path) as f:
            result["feature_importance"] = json.load(f)
    
    return result