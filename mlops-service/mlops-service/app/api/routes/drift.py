from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from app.api.dependencies import get_settings
from app.config.settings import Settings
from app.core.exceptions import MLOpsException
from app.platform.event_client import send_raw_event
from app.monitoring.drift_detection import (
    DRIFT_THRESHOLD_PSI,
    DriftDetectionService,
    DriftStatus,
)
from app.monitoring.prometheus_metrics import (
    DRIFT_DETECTED,
    DRIFT_PSI_SCORE,
    sanitize_label_value,
)


class DriftCheckRequest(BaseModel):
    reference_path: str
    current_path: str
    pipeline: str = "imaging"


class DriftCheckResponse(BaseModel):
    pipeline: str
    status: str
    overall_psi: float
    drift_detected: bool
    reference_samples: int
    current_samples: int
    generated_at: str
    threshold_psi: float = DRIFT_THRESHOLD_PSI


class DriftStatusResponse(BaseModel):
    pipeline: str
    status: str
    overall_psi: float
    drift_detected: bool
    psi_threshold: float = DRIFT_THRESHOLD_PSI
    features_drifted: list[str] = []
    last_checked: Optional[str] = None


class DriftHistoryResponse(BaseModel):
    history: list[dict]
    total: int


router = APIRouter(prefix="/drift", tags=["drift"])


def get_drift_service(
    settings: Settings = Depends(get_settings),
) -> DriftDetectionService:
    artifacts_root = settings.artifacts_root
    reports_dir = artifacts_root / "monitoring" / "drift"
    return DriftDetectionService(artifacts_root, reports_dir)


@router.post("/check", response_model=DriftCheckResponse)
async def check_drift(
    request: DriftCheckRequest,
    settings: Settings = Depends(get_settings),
    service: DriftDetectionService = Depends(get_drift_service),
) -> DriftCheckResponse:
    reference_path = Path(request.reference_path)
    current_path = Path(request.current_path)

    if not reference_path.is_absolute():
        reference_path = settings.artifacts_root / reference_path
    if not current_path.is_absolute():
        current_path = settings.artifacts_root / current_path

    try:
        report = service.check_drift(
            reference_csv=reference_path,
            current_csv=current_path,
            pipeline=request.pipeline,
        )

        DRIFT_DETECTED.labels(pipeline=request.pipeline).set(
            1 if report.drift_detected else 0
        )

        for feature_result in report.feature_results:
            DRIFT_PSI_SCORE.labels(
                pipeline=request.pipeline,
                feature=sanitize_label_value(feature_result.feature_name),
            ).set(feature_result.psi)

        if report.drift_detected:
            await send_raw_event(
                event="drift.detected",
                data={
                    "pipeline": request.pipeline,
                    "overall_psi": report.overall_psi,
                    "threshold": DRIFT_THRESHOLD_PSI,
                    "status": report.status,
                },
                room="drift",
            )

        return DriftCheckResponse(
            pipeline=report.pipeline,
            status=report.status,
            overall_psi=report.overall_psi,
            drift_detected=report.drift_detected,
            reference_samples=report.reference_samples,
            current_samples=report.current_samples,
            generated_at=report.generated_at,
        )

    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise MLOpsException(
            status_code=500, detail=str(e), error_code="DRIFT_CHECK_ERROR"
        )


@router.get("/status/{pipeline}", response_model=DriftStatusResponse)
async def get_drift_status(
    pipeline: str,
    service: DriftDetectionService = Depends(get_drift_service),
) -> DriftStatusResponse:
    latest = service.get_latest_drift(pipeline)

    features_drifted = []
    if latest and latest.feature_results:
        features_drifted = [
            f.feature_name for f in latest.feature_results if f.drift_detected
        ]

    return DriftStatusResponse(
        pipeline=pipeline,
        status=latest.status if latest else DriftStatus.UNKNOWN,
        overall_psi=latest.overall_psi if latest else 0.0,
        drift_detected=latest.drift_detected if latest else False,
        features_drifted=features_drifted,
        last_checked=latest.generated_at if latest else None,
    )


@router.get("/history", response_model=DriftHistoryResponse)
async def get_drift_history(
    pipeline: Optional[str] = None,
    limit: int = 10,
    service: DriftDetectionService = Depends(get_drift_service),
) -> DriftHistoryResponse:
    history = service.get_drift_history(pipeline=pipeline, limit=limit)

    return DriftHistoryResponse(
        history=history,
        total=len(history),
    )
