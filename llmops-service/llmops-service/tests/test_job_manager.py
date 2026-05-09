from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services.job_manager import JobManager, JobStatus


class TestJobManager:
    @pytest.fixture
    def manager(self, tmp_path: Path) -> JobManager:
        return JobManager(max_concurrent=2, persist_dir=tmp_path / "jobs")

    @pytest.fixture
    def dummy_handler(self):
        async def handler(job):
            return {"result": "ok"}

        return handler

    @pytest.mark.asyncio
    async def test_worker_loop_processes_job(self, manager: JobManager, dummy_handler):
        manager.register_handler("test", dummy_handler)
        await manager.start()

        job_id = await manager.submit("test", {"key": "value"})
        await asyncio.sleep(0.3)

        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"result": "ok"}

        await manager.stop()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, manager: JobManager):
        call_count = 0

        async def failing_handler(job):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        manager.register_handler("fail", failing_handler)
        await manager.start()

        job_id = await manager.submit("fail", {}, max_retries=2)
        await asyncio.sleep(7.0)

        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.retry_count == 2
        assert call_count == 2

        await manager.stop()

    @pytest.mark.asyncio
    async def test_persistence(
        self, manager: JobManager, dummy_handler, tmp_path: Path
    ):
        manager.register_handler("persist", dummy_handler)
        await manager.start()

        job_id = await manager.submit("persist", {"data": "test"})
        await asyncio.sleep(0.3)
        await manager.stop()

        persist_file = tmp_path / "jobs" / f"{job_id}.json"
        assert persist_file.exists()

        data = json.loads(persist_file.read_text())
        assert data["job_type"] == "persist"
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_job_eviction_cancel(self, manager: JobManager, dummy_handler):
        manager.register_handler("cancel", dummy_handler)
        await manager.start()

        job_id = await manager.submit("cancel", {})
        success = await manager.cancel_job(job_id)
        assert success is True

        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.CANCELLED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_cancel_completed_job_fails(self, manager: JobManager, dummy_handler):
        manager.register_handler("done", dummy_handler)
        await manager.start()

        job_id = await manager.submit("done", {})
        await asyncio.sleep(0.3)

        success = await manager.cancel_job(job_id)
        assert success is False

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_jobs_filter(self, manager: JobManager, dummy_handler):
        manager.register_handler("filter", dummy_handler)
        await manager.start()

        job_id = await manager.submit("filter", {})
        await asyncio.sleep(0.3)

        completed = manager.get_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == job_id

        pending = manager.get_jobs(status=JobStatus.PENDING)
        assert len(pending) == 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_load_persisted_jobs(self, manager: JobManager, tmp_path: Path):
        async def slow_handler(job):
            await asyncio.sleep(10)
            return {"result": "slow"}

        manager.register_handler("restore", slow_handler)
        await manager.start()

        job_id = await manager.submit("restore", {"restore": True})
        await asyncio.sleep(0.1)
        await manager.stop()

        new_manager = JobManager(max_concurrent=2, persist_dir=tmp_path / "jobs")
        new_manager.register_handler("restore", slow_handler)
        await new_manager.start()
        await asyncio.sleep(0.2)

        job = new_manager.get_job(job_id)
        assert job is not None
        assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

        await new_manager.stop()
