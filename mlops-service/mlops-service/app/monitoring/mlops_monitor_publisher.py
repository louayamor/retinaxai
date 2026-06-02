from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from redis.asyncio import Redis
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.api.routes.metrics import get_metrics
from app.api.dependencies import get_settings


class _MetricsUpdateHandler(FileSystemEventHandler):
    def __init__(self, publisher: "MLOpsMonitorPublisher") -> None:
        self.publisher = publisher

    def on_modified(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self.publisher.schedule_publish(Path(event.src_path))

    def on_created(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self.publisher.schedule_publish(Path(event.src_path))


class MLOpsMonitorPublisher:
    def __init__(self, redis_url: str, channel: str) -> None:
        self._redis_url = redis_url
        self._channel = channel
        self._observer: Observer | None = None
        self._pending: concurrent.futures.Future | None = None
        self._lock = asyncio.Lock()
        self._watched_files: set[Path] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, watch_paths: list[Path]) -> None:
        self._loop = asyncio.get_running_loop()
        handler = _MetricsUpdateHandler(self)
        observer = Observer()
        for path in watch_paths:
            parent = path.parent
            if not parent.exists():
                logger.warning("mlops_monitor_skip_missing_dir", path=str(parent))
                continue
            self._watched_files.add(path.resolve())
            observer.schedule(handler, str(parent), recursive=False)
        if not self._watched_files:
            logger.info("mlops_monitor_no_paths_to_watch")
            self._observer = None
            return
        observer.start()
        self._observer = observer
        logger.info(
            "mlops_monitor_watch_started",
            channel=self._channel,
            files=[str(p) for p in self._watched_files],
        )

    async def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._pending:
            self._pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wrap_future(self._pending)

    def schedule_publish(self, changed_path: Path) -> None:
        if changed_path.resolve() not in self._watched_files:
            return
        if self._pending and not self._pending.done():
            return
        if self._loop is None:
            return
        coro = self.publish_snapshot()
        self._pending = asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def publish_snapshot(self) -> None:
        async with self._lock:
            payload = build_monitor_snapshot()
            serialized = json.dumps(payload, separators=(",", ":"))
            try:
                async with Redis.from_url(self._redis_url) as redis:
                    await redis.publish(self._channel, serialized)
                logger.info("mlops_monitor_published", channel=self._channel)
            except Exception as exc:
                logger.warning(
                    "mlops_monitor_publish_failed",
                    error=str(exc),
                    channel=self._channel,
                )


def build_monitor_snapshot() -> dict[str, Any]:
    metrics = get_metrics()
    metrics_data: dict[str, Any] = metrics.model_dump()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "imaging": metrics_data.get("imaging"),
            "clinical": metrics_data.get("clinical"),
        },
        "training_summary": metrics_data.get("training_summary"),
        "prometheus": {},
    }
