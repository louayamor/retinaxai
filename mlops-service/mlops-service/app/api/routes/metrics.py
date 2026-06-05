import json
from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsResponse, ImagingMetrics
from app.api.dependencies import get_settings
from app.model.loader import read_json_from_gcs

router = APIRouter()

GCS_ARTIFACTS_PREFIX = "imaging"


def _read_artifact_json(settings, local_path, gcs_filename: str):
    """Read a JSON artifact: try GCS first, fall back to local file."""
    bucket = settings.gcs_model_bucket
    if bucket:
        key = f"{GCS_ARTIFACTS_PREFIX}/{gcs_filename}"
        data = read_json_from_gcs(bucket, key)
        if data is not None:
            return data
    if local_path.is_file():
        with open(local_path) as f:
            return json.load(f)
    return None


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    settings = get_settings()

    data = _read_artifact_json(
        settings, settings.imaging_metrics_path, "metrics.json"
    )

    imaging_metrics = None
    imaging_detail = None
    if data is not None:
        eyepacs = data.get("eyepacs_test", {})
        imaging_metrics = ImagingMetrics(
            accuracy=eyepacs.get("accuracy"),
            quadratic_weighted_kappa=eyepacs.get("quadratic_weighted_kappa"),
            roc_auc_macro=eyepacs.get("roc_auc_macro"),
            precision_macro=eyepacs.get("precision_macro"),
            recall_macro=eyepacs.get("recall_macro"),
            num_samples=eyepacs.get("num_samples"),
        )
        imaging_detail = data

    training_summary = _read_artifact_json(
        settings,
        settings.imaging_artifacts_dir / "training_summary.json",
        "training_summary.json",
    )

    return MetricsResponse(
        imaging=imaging_metrics,
        imaging_detail=imaging_detail,
        training_summary=training_summary,
    )
