from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.dependencies import get_settings

router = APIRouter(prefix="/models", tags=["models"])

ALLOWED_ARTIFACT_EXTENSIONS = {".png", ".json", ".csv", ".jsonl"}


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
        raise HTTPException(
            status_code=400, detail="pipeline must be 'clinical' or 'imaging'"
        )

    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model not found for pipeline: {pipeline}"
        )

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
        raise HTTPException(
            status_code=400, detail="pipeline must be 'clinical' or 'imaging'"
        )

    metadata_dir = settings.artifacts_root / "model" / pipeline
    if not metadata_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Model metadata not found for: {pipeline}"
        )

    result = {"pipeline": pipeline, "files": []}

    for f in metadata_dir.glob("*"):
        if f.is_file() and f.suffix in [".json", ".pkl", ".pth"]:
            result["files"].append(
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                }
            )

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


@router.get("/download/{pipeline}/artifacts/{filename}")
async def download_artifact(
    pipeline: str,
    filename: str,
) -> FileResponse:
    """Serve a model artifact file (confusion matrix PNG, etc.).

    Only files with allowed extensions in the pipeline directory are served.
    """
    settings = get_settings()

    if pipeline not in ["clinical", "imaging"]:
        raise HTTPException(
            status_code=400, detail="pipeline must be 'clinical' or 'imaging'"
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_ARTIFACT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{suffix}' not allowed")

    file_path = (settings.artifacts_root / "model" / pipeline / filename).resolve()
    artifacts_dir = (settings.artifacts_root / "model" / pipeline).resolve()

    if not str(file_path).startswith(str(artifacts_dir)):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")

    media_type = "image/png" if suffix == ".png" else "application/json"
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )


@router.get("/download/{pipeline}/misclassified/{split}")
async def list_misclassified(
    pipeline: str,
    split: str,
) -> dict:
    """List misclassified image filenames and count for a given split."""
    settings = get_settings()

    if pipeline not in ["clinical", "imaging"]:
        raise HTTPException(
            status_code=400, detail="pipeline must be 'clinical' or 'imaging'"
        )

    misclassified_dir = (
        settings.artifacts_root / "model" / pipeline / "misclassified" / split
    )
    if not misclassified_dir.is_dir():
        raise HTTPException(
            status_code=404, detail=f"Misclassified split not found: {split}"
        )

    files = sorted(
        [
            f.name
            for f in misclassified_dir.iterdir()
            if f.is_file() and f.suffix == ".png"
        ]
    )

    return {
        "pipeline": pipeline,
        "split": split,
        "count": len(files),
        "files": files,
    }


@router.get("/download/{pipeline}/misclassified/{split}/{filename}")
async def download_misclassified_image(
    pipeline: str,
    split: str,
    filename: str,
) -> FileResponse:
    """Serve a single misclassified image."""
    settings = get_settings()

    if pipeline not in ["clinical", "imaging"]:
        raise HTTPException(
            status_code=400, detail="pipeline must be 'clinical' or 'imaging'"
        )

    if not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only .png files are served")

    file_path = (
        settings.artifacts_root
        / "model"
        / pipeline
        / "misclassified"
        / split
        / filename
    ).resolve()
    misclassified_dir = (
        settings.artifacts_root / "model" / pipeline / "misclassified" / split
    ).resolve()

    if not str(file_path).startswith(str(misclassified_dir)):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename,
    )
