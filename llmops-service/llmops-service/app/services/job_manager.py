"""
Async Job Manager for LLMOps Service.

Manages report generation jobs with status tracking, retries, and persistence.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from app.core.config import settings
from app.services.operation_state import get_operation_state_manager

MAX_JOBS = 1000


class JobStatus(StrEnum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a report generation job."""

    id: str
    job_type: str
    payload: dict[str, Any]
    status: JobStatus = field(default=JobStatus.PENDING)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Create job from dictionary."""
        return cls(
            id=data["id"],
            job_type=data["job_type"],
            payload=data["payload"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )


class JobManager:
    """
    Manages async report generation jobs.

    Features:
    - In-memory job queue with persistence
    - Status tracking (pending, running, completed, failed)
    - Retry mechanism with exponential backoff
    - Concurrent job execution
    """

    def __init__(self, max_concurrent: int = 5, persist_dir: Path | None = None):
        self.max_concurrent = max_concurrent
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._workers: list[asyncio.Task] = []
        self._running = False

        # Persistence
        self._persist_dir = Path(persist_dir or settings.cache_dir / "jobs")
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Job handlers
        self._handlers: dict[
            str, Callable[[Job], Coroutine[Any, Any, dict[str, Any]]]
        ] = {}

    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Job], Coroutine[Any, Any, dict[str, Any]]],
    ) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    async def start(self) -> None:
        """Start the job manager workers."""
        if self._running:
            return

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker_loop(), name=f"job-worker-{i}")
            for i in range(self.max_concurrent)
        ]

        # Load persisted jobs
        await self._load_jobs()

        logger.info(f"Job manager started with {self.max_concurrent} workers")

    async def stop(self) -> None:
        """Stop the job manager gracefully."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        # Persist jobs
        await self._persist_jobs()

        logger.info("Job manager stopped")

    def _evict_if_needed(self) -> None:
        """Evict oldest completed/failed jobs if MAX_JOBS exceeded."""
        if len(self.jobs) <= MAX_JOBS:
            return

        evictable = [
            (job_id, job)
            for job_id, job in self.jobs.items()
            if job.status
            in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        ]
        evictable.sort(key=lambda item: item[1].completed_at or item[1].created_at)

        excess = len(self.jobs) - MAX_JOBS
        to_evict = evictable[:excess]

        for job_id, _ in to_evict:
            del self.jobs[job_id]
            try:
                job_file = self._persist_dir / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove persisted job file {job_id}: {e}")

        if to_evict:
            logger.info(
                f"Evicted {len(to_evict)} old jobs to stay within MAX_JOBS={MAX_JOBS}"
            )

    async def submit(
        self, job_type: str, payload: dict[str, Any], max_retries: int = 3
    ) -> str:
        """
        Submit a new job.

        Args:
            job_type: Type of job (e.g., "report_generation")
            payload: Job input data
            max_retries: Maximum retry attempts

        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type=job_type,
            payload=payload,
            max_retries=max_retries,
        )

        self.jobs[job_id] = job
        await self._queue.put(job_id)
        self._evict_if_needed()

        logger.info(f"Job {job_id} submitted (type: {job_type})")
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def get_jobs(self, status: JobStatus | None = None, limit: int = 100) -> list[Job]:
        """Get jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.RETRYING):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            await self._persist_job(job)
            logger.info(f"Job {job_id} cancelled")
            return True

        return False

    async def _worker_loop(self) -> None:
        """Worker loop that processes jobs from the queue."""
        while self._running:
            try:
                # Get job from queue with timeout
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_job(job_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

    async def _process_job(self, job_id: str) -> None:
        """Process a single job."""
        async with self._semaphore:
            job = self.jobs.get(job_id)
            if not job or job.status not in (
                JobStatus.PENDING,
                JobStatus.RETRYING,
            ):
                return

            handler = self._handlers.get(job.job_type)
            if not handler:
                job.status = JobStatus.FAILED
                job.error = f"No handler for job type: {job.job_type}"
                await self._persist_job(job)
                return

            op_manager = get_operation_state_manager()
            op_manager.set_operation("generating", f"Processing {job.job_type}...")

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            await self._persist_job(job)

            try:
                result = await handler(job)
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = datetime.now(timezone.utc)
                logger.info(f"Job {job_id} completed successfully")

            except Exception as e:
                job.retry_count += 1

                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    job.error = str(e)
                    logger.warning(
                        f"Job {job_id} failed (attempt {job.retry_count}), retrying: {e}"
                    )
                    # Re-queue for retry with backoff
                    await asyncio.sleep(2**job.retry_count)  # Exponential backoff
                    await self._queue.put(job_id)
                else:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now(timezone.utc)
                    logger.error(
                        f"Job {job_id} failed after {job.max_retries} retries: {e}"
                    )

            finally:
                op_manager = get_operation_state_manager()
                if job.status == JobStatus.FAILED:
                    error_msg = job.error or "Unknown error"
                    op_manager.set_operation("error", error_msg[:200])
                elif job.status == JobStatus.COMPLETED:
                    op_manager.set_operation("idle", "Ready")
                # Don't clear state for PENDING/RETRYING/RUNNING - keep showing job progress
                await self._persist_job(job)
                self._evict_if_needed()

    async def _persist_job(self, job: Job) -> None:
        """Persist a single job to disk."""
        try:
            job_file = self._persist_dir / f"{job.id}.json"
            job_file.write_text(
                json.dumps(job.to_dict(), default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to persist job {job.id}: {e}")

    async def _persist_jobs(self) -> None:
        """Persist all jobs to disk."""
        for job in self.jobs.values():
            await self._persist_job(job)

    async def _load_jobs(self) -> None:
        """Load persisted jobs from disk."""
        try:
            for job_file in self._persist_dir.glob("*.json"):
                try:
                    data = json.loads(job_file.read_text(encoding="utf-8"))
                    job = Job.from_dict(data)

                    # Load all jobs, but only re-queue incomplete ones
                    self.jobs[job.id] = job
                    if job.status in (
                        JobStatus.PENDING,
                        JobStatus.RUNNING,
                        JobStatus.RETRYING,
                    ):
                        job.status = JobStatus.PENDING  # Reset to pending
                        await self._queue.put(job.id)
                        logger.info(f"Restored job {job.id} from persistence")
                    else:
                        logger.info(f"Loaded completed job {job.id} from persistence")
                except Exception as e:
                    logger.warning(f"Failed to load job from {job_file}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load jobs: {e}")


_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """FastAPI dependency factory. Creates instance if not overridden."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
