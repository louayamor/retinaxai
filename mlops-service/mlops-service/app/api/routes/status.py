from fastapi import APIRouter
from app.api.schemas import StatusResponse
from app.core.exceptions import NotFoundException
from app.training.orchestration.training_service import get_job_status

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    job = get_job_status(job_id)
    if not job:
        raise NotFoundException("Job", job_id)
    return StatusResponse(**job)


@router.get("/status", response_model=StatusResponse)
def get_latest_status():
    from app.training.orchestration.training_service import get_latest_job

    latest = get_latest_job()
    if not latest:
        return StatusResponse(
            job_id=None,
            pipeline=None,
            status="idle",
            started_at=None,
            completed_at=None,
            error=None,
        )
    return StatusResponse(**latest)
