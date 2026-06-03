from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.auth.role_guard import EngineerUser
from app.services.event_queue import EventStatus

router = APIRouter(prefix="/v1/orchestration", tags=["orchestration"])


class OrchestrationStatus(BaseModel):
    status: str
    connected_clients: int
    queued_events: int
    active_predictions: int
    active_training: int
    llmops_available: bool
    mlops_available: bool
    timestamp: str


class WorkflowInfo(BaseModel):
    id: str
    type: str
    status: str
    created_at: str | None
    completed_at: str | None
    error: str | None


class WorkflowHistory(BaseModel):
    workflows: list[WorkflowInfo]
    total: int


MLOPS_AVAILABLE = True
LLMOPS_AVAILABLE = True


def _get_training_jobs() -> dict:
    """Get training jobs from MLOps."""
    try:
        mlops_jobs_file = Path(
            "mlops-service/mlops-service/artifacts/training_jobs.json"
        )
        if mlops_jobs_file.exists():
            with open(mlops_jobs_file) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load training jobs: {e}")
    return {}


def _get_queued_events() -> list[dict]:
    """Get queued events from event queue."""
    try:
        event_queue_file = Path("backend-service/data/event_queue.json")
        if event_queue_file.exists():
            with open(event_queue_file) as f:
                data = json.load(f)
                return data.get("queue", [])
    except Exception as e:
        logger.warning(f"Failed to load event queue: {e}")
    return []


@router.get("/status", response_model=OrchestrationStatus)
async def get_orchestration_status(_: EngineerUser) -> OrchestrationStatus:
    """
    Get overall orchestration health status.
    """
    training_jobs = _get_training_jobs()
    queued_events = _get_queued_events()

    active_training = sum(
        1
        for job in training_jobs.values()
        if job.get("status") in ("running", "pending")
    )

    from app.api.v1.websockets import _connected_clients

    mlops_status = _check_service_health("http://localhost:8004/health")
    llmops_status = _check_service_health("http://localhost:8002/health")

    return OrchestrationStatus(
        status="healthy" if (mlops_status and llmops_status) else "degraded",
        connected_clients=len(_connected_clients),
        queued_events=len(queued_events),
        active_predictions=0,
        active_training=active_training,
        llmops_available=llmops_status,
        mlops_available=mlops_status,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/workflows", response_model=WorkflowHistory)
async def get_active_workflows(
    _: EngineerUser,
    status: str | None = None,
    limit: int = 50,
) -> WorkflowHistory:
    """
    Get list of active workflows (training jobs and queued events).
    """
    workflows: list[WorkflowInfo] = []

    training_jobs = _get_training_jobs()
    for job_id, job in training_jobs.items():
        job_status = job.get("status", "unknown")
        if status and job_status != status:
            continue

        workflows.append(
            WorkflowInfo(
                id=job_id,
                type="training",
                status=job_status,
                created_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                error=job.get("error"),
            )
        )

    queued_events = _get_queued_events()
    for event in queued_events:
        event_status = event.get("status", "unknown")
        if status and event_status != status:
            continue

        workflows.append(
            WorkflowInfo(
                id=event.get("id", ""),
                type="event",
                status=event_status,
                created_at=event.get("created_at"),
                completed_at=event.get("last_retry_at"),
                error=event.get("last_error"),
            )
        )

    workflows.sort(key=lambda w: w.created_at or "", reverse=True)

    return WorkflowHistory(
        workflows=workflows[:limit],
        total=len(workflows),
    )


@router.get("/history", response_model=WorkflowHistory)
async def get_workflow_history(
    _: EngineerUser,
    limit: int = 100,
    status: str | None = None,
    workflow_type: str | None = None,
) -> WorkflowHistory:
    """
    Get workflow history (completed and failed workflows).
    """
    workflows: list[WorkflowInfo] = []

    training_jobs = _get_training_jobs()
    for job_id, job in training_jobs.items():
        job_status = job.get("status", "unknown")

        if workflow_type is not None and workflow_type != "training":
            continue

        if job_status not in ("completed", "failed", "cancelled"):
            continue

        workflows.append(
            WorkflowInfo(
                id=job_id,
                type="training",
                status=job_status,
                created_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                error=job.get("error"),
            )
        )

    workflows.sort(key=lambda w: w.completed_at or w.created_at or "", reverse=True)

    return WorkflowHistory(
        workflows=workflows[:limit],
        total=len(workflows),
    )


@router.delete("/queue/{event_id}")
async def cancel_queued_event(
    _: EngineerUser,
    event_id: str,
) -> dict:
    """
    Cancel a queued event by removing it from the queue.
    """
    try:
        event_queue_file = Path("backend-service/data/event_queue.json")
        if not event_queue_file.exists():
            raise HTTPException(status_code=404, detail="Event queue not found")

        with open(event_queue_file) as f:
            data = json.load(f)

        queue = data.get("queue", [])
        original_length = len(queue)
        queue = [e for e in queue if e.get("id") != event_id]

        if len(queue) == original_length:
            raise HTTPException(status_code=404, detail="Event not found")

        data["queue"] = queue
        with open(event_queue_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Cancelled queued event: {event_id}")
        return {"status": "ok", "event_id": event_id, "message": "Event cancelled"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retry/{event_id}")
async def retry_event(
    _: EngineerUser,
    event_id: str,
) -> dict:
    """
    Manually retry a failed event.
    """
    try:
        from app.services.event_queue import get_event_queue

        event_queue = get_event_queue()

        for event in event_queue.get_pending_events():
            if event.id == event_id:
                event.status = EventStatus.PENDING  # type: ignore[assignment]
                event.retry_count = 0
                logger.info(f"Reset event for retry: {event_id}")
                return {
                    "status": "ok",
                    "event_id": event_id,
                    "message": "Event queued for retry",
                }

        raise HTTPException(status_code=404, detail="Event not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _check_service_health(url: str) -> bool:
    """Check if a service is available."""
    try:
        import httpx
        import asyncio

        async def check():
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                return response.status_code < 400

        return asyncio.run(check())
    except Exception:
        return False
