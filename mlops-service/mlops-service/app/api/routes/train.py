from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from app.api.schemas import TrainRequest, TrainResponse
from app.api.dependencies import get_settings
from app.config.settings import Settings
from app.services.orchestration.training_service import (
    create_job,
    run_pipeline_task,
    cancel_job,
    _job_store,
)
from app.services.platform.resource_manager import ResourceManager

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
def trigger_full_pipeline(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    gate = ResourceManager(
        max_jobs=settings.max_training_jobs,
        max_jobs_per_pipeline=settings.max_training_jobs_per_pipeline,
    ).can_start(request.pipeline.value)
    if not gate.allowed:
        raise HTTPException(status_code=429, detail=f"Training rejected: {gate.reason}")
    job_id = create_job(request.pipeline.value)
    background_tasks.add_task(run_pipeline_task, job_id, request.pipeline.value)
    return TrainResponse(
        job_id=job_id,
        pipeline=request.pipeline.value,
        status="pending",
        message=f"training job queued for pipeline: {request.pipeline.value}",
    )


@router.post("/train/imaging", response_model=TrainResponse)
def trigger_imaging_pipeline(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    gate = ResourceManager(
        max_jobs=settings.max_training_jobs,
        max_jobs_per_pipeline=settings.max_training_jobs_per_pipeline,
    ).can_start("imaging")
    if not gate.allowed:
        raise HTTPException(status_code=429, detail=f"Training rejected: {gate.reason}")
    job_id = create_job("imaging")
    background_tasks.add_task(run_pipeline_task, job_id, "imaging")
    return TrainResponse(
        job_id=job_id,
        pipeline="imaging",
        status="pending",
        message="imaging training job queued",
    )


@router.post("/train/clinical", response_model=TrainResponse)
def trigger_clinical_pipeline(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    gate = ResourceManager(
        max_jobs=settings.max_training_jobs,
        max_jobs_per_pipeline=settings.max_training_jobs_per_pipeline,
    ).can_start("clinical")
    if not gate.allowed:
        raise HTTPException(status_code=429, detail=f"Training rejected: {gate.reason}")
    job_id = create_job("clinical")
    background_tasks.add_task(run_pipeline_task, job_id, "clinical")
    return TrainResponse(
        job_id=job_id,
        pipeline="clinical",
        status="pending",
        message="clinical training job queued",
    )


@router.post("/train/{job_id}/stop")
def stop_training(job_id: str):
    """Stop a running training job."""
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"message": f"Job {job_id} stop requested"}
