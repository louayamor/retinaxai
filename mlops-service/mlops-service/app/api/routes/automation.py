from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_settings
from app.config.settings import Settings
from app.services.orchestration.automation_service import get_automation_service


class AutomationStatusResponse(BaseModel):
    enabled: bool
    interval_hours: int


class DriftRetrainRequest(BaseModel):
    pipeline: str = "both"
    psi_threshold: float = 0.3


class DriftRetrainResponse(BaseModel):
    status: str
    psi: float
    job_id: Optional[str] = None


router = APIRouter(prefix="/automation", tags=["automation"])


def _get_service(settings: Settings):
    artifacts_root = settings.artifacts_root
    reports_dir = artifacts_root / "monitoring" / "drift"
    return get_automation_service(artifacts_root, reports_dir)


@router.post("/start")
async def start_automation(
    settings: Settings = Depends(get_settings),
) -> AutomationStatusResponse:
    if not settings.automation_enabled:
        raise HTTPException(status_code=400, detail="automation is disabled")

    service = _get_service(settings)
    service.start_scheduler(interval_hours=settings.automation_interval_hours)

    return AutomationStatusResponse(
        enabled=True,
        interval_hours=settings.automation_interval_hours,
    )


@router.post("/stop")
async def stop_automation(
    settings: Settings = Depends(get_settings),
) -> AutomationStatusResponse:
    service = _get_service(settings)
    service.stop_scheduler()

    return AutomationStatusResponse(
        enabled=False,
        interval_hours=settings.automation_interval_hours,
    )


@router.post("/drift-retrain", response_model=DriftRetrainResponse)
async def drift_retrain(
    request: DriftRetrainRequest,
    settings: Settings = Depends(get_settings),
) -> DriftRetrainResponse:
    service = _get_service(settings)

    if request.pipeline in ("imaging", "both"):
        reference_path = settings.imaging_train_csv
        current_path = settings.imaging_test_csv
    else:
        reference_path = settings.clinical_train_csv
        current_path = settings.clinical_test_csv

    result = service.trigger_drift_retraining(
        reference_csv=reference_path,
        current_csv=current_path,
        pipeline=request.pipeline,
        psi_threshold=request.psi_threshold,
    )

    return DriftRetrainResponse(
        status=result["status"],
        psi=result["psi"],
        job_id=result.get("job_id"),
    )
