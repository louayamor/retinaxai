from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.dependencies import get_settings
from app.core.exceptions import (
    ForbiddenException,
    NotFoundException,
    UnprocessableEntityException,
)

router = APIRouter(prefix="/models", tags=["models"])

ALLOWED_ARTIFACT_EXTENSIONS = {".png", ".json", ".csv", ".jsonl"}


def _validate_pipeline(pipeline: str) -> None:
    if pipeline not in ["imaging"]:
        raise UnprocessableEntityException("pipeline must be 'imaging'")


@router.get("/download/{pipeline}/model")
async def download_model(
    pipeline: str,
) -> FileResponse:
    """Download the current production model for the imaging pipeline."""
    settings = get_settings()
    _validate_pipeline(pipeline)

    model_path = settings.artifacts_root / "model" / "imaging" / "model.pth"
    media_type = "application/octet-stream"

    if not model_path.exists():
        raise NotFoundException("Model", pipeline)

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
    _validate_pipeline(pipeline)

    metadata_dir = settings.artifacts_root / "model" / pipeline
    if not metadata_dir.exists():
        raise NotFoundException("Model metadata", pipeline)

    result: dict = {"pipeline": pipeline, "files": []}

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
    _validate_pipeline(pipeline)

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_ARTIFACT_EXTENSIONS:
        raise UnprocessableEntityException(f"File type '{suffix}' not allowed")

    file_path = (settings.artifacts_root / "model" / pipeline / filename).resolve()
    artifacts_dir = (settings.artifacts_root / "model" / pipeline).resolve()

    if not str(file_path).startswith(str(artifacts_dir)):
        raise ForbiddenException("Path traversal not allowed")

    if not file_path.is_file():
        raise NotFoundException("Artifact", filename)

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
    _validate_pipeline(pipeline)

    misclassified_dir = (
        settings.artifacts_root / "model" / pipeline / "misclassified" / split
    )
    if not misclassified_dir.is_dir():
        raise NotFoundException("Misclassified split", split)

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
    _validate_pipeline(pipeline)

    if not filename.endswith(".png"):
        raise UnprocessableEntityException("Only .png files are served")

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
        raise ForbiddenException("Path traversal not allowed")

    if not file_path.is_file():
        raise NotFoundException("Image", filename)

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename,
    )
