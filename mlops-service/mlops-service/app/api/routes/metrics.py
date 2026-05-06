import json
from fastapi import APIRouter, HTTPException
from app.api.schemas import MetricsResponse, ImagingMetrics, ClinicalMetrics
from app.api.dependencies import get_settings

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    settings = get_settings()

    imaging_metrics = None
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

    clinical_metrics = None
    if settings.clinical_metrics_path.is_file():
        with open(settings.clinical_metrics_path) as f:
            data = json.load(f)
        clinical_metrics = ClinicalMetrics(
            accuracy=data.get("accuracy"),
            quadratic_weighted_kappa=data.get("quadratic_weighted_kappa"),
            roc_auc_macro=data.get("roc_auc_macro"),
            precision_macro=data.get("precision_macro"),
            recall_macro=data.get("recall_macro"),
            num_samples=data.get("num_samples"),
        )

    return MetricsResponse(imaging=imaging_metrics, clinical=clinical_metrics)
