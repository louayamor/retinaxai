import json
from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsResponse, ImagingMetrics
from app.api.dependencies import get_settings

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    settings = get_settings()

    imaging_metrics = None
    imaging_detail = None
    if settings.imaging_metrics_path.is_file():
        with open(settings.imaging_metrics_path) as f:
            data = json.load(f)
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

    training_summary = None
    training_summary_path = settings.imaging_artifacts_dir / "training_summary.json"
    if training_summary_path.is_file():
        with open(training_summary_path) as f:
            training_summary = json.load(f)

    return MetricsResponse(
        imaging=imaging_metrics,
        imaging_detail=imaging_detail,
        training_summary=training_summary,
    )
