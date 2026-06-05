from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from app.api.schemas import TrainRequest, TrainResponse
from app.api.dependencies import get_settings
from app.config.settings import Settings
from app.core.exceptions import MLOpsException, NotFoundException
from app.platform.event_client import send_raw_event
from app.training.orchestration.training_service import (
    create_job,
    run_pipeline_task,
    cancel_job,
    _job_store,
)
from app.platform.resource_manager import ResourceManager

router = APIRouter(prefix="/api")


class JobListResponse(BaseModel):
    jobs: list[dict]
    total: int


@router.get("/train/jobs", response_model=JobListResponse)
def list_training_jobs():
    jobs = list(_job_store.values())
    jobs.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.post("/train", response_model=TrainResponse)
async def trigger_imaging_pipeline(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    gate = ResourceManager(
        max_jobs=settings.max_training_jobs,
        max_jobs_per_pipeline=settings.max_training_jobs_per_pipeline,
    ).can_start("imaging")
    if not gate.allowed:
        await send_raw_event(
            event="training.rejected",
            data={
                "pipeline": "imaging",
                "reason": gate.reason,
            },
            room="training:imaging",
        )
        raise MLOpsException(
            status_code=429,
            detail=f"Training rejected: {gate.reason}",
            error_code="TRAINING_REJECTED",
        )
    job_id = create_job("imaging")
    await send_raw_event(
        event="training.queued",
        data={
            "job_id": job_id,
            "pipeline": "imaging",
            "status": "queued",
        },
        room="training:imaging",
    )
    background_tasks.add_task(run_pipeline_task, job_id, "imaging")
    return TrainResponse(
        job_id=job_id,
        pipeline="imaging",
        status="pending",
        message="imaging training job queued",
    )


@router.post("/train/fundus", response_model=TrainResponse)
async def trigger_fundus_pipeline(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    gate = ResourceManager(
        max_jobs=settings.max_training_jobs,
        max_jobs_per_pipeline=settings.max_training_jobs_per_pipeline,
    ).can_start("fundus")
    if not gate.allowed:
        await send_raw_event(
            event="training.rejected",
            data={
                "pipeline": "fundus",
                "reason": gate.reason,
            },
            room="training:fundus",
        )
        raise MLOpsException(
            status_code=429,
            detail=f"Training rejected: {gate.reason}",
            error_code="TRAINING_REJECTED",
        )
    job_id = create_job("fundus")
    await send_raw_event(
        event="training.queued",
        data={
            "job_id": job_id,
            "pipeline": "fundus",
            "status": "queued",
        },
        room="training:fundus",
    )
    background_tasks.add_task(run_pipeline_task, job_id, "fundus")
    return TrainResponse(
        job_id=job_id,
        pipeline="fundus",
        status="pending",
        message="fundus classifier training job queued",
    )


@router.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a running training job."""
    success = cancel_job(job_id)
    if not success:
        raise NotFoundException("Training job", job_id)
    await send_raw_event(
        event="training.cancelled",
        data={
            "job_id": job_id,
            "pipeline": "imaging",
            "status": "cancelled",
        },
        room="training:imaging",
    )
    return {"message": f"Job {job_id} stop requested"}
