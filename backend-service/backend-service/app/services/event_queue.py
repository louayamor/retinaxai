from __future__ import annotations
import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class EventStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class QueuedEvent:
    id: str
    event: str
    data: dict[str, Any]
    room: str | None
    created_at: str
    retry_count: int = 0
    max_retries: int = 5
    status: EventStatus = EventStatus.PENDING
    last_error: str | None = None
    last_retry_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event": self.event,
            "data": self.data,
            "room": self.room,
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "last_error": self.last_error,
            "last_retry_at": self.last_retry_at,
        }


class EventQueueService:
    """
    Service for queuing events when services are unavailable.
    Implements retry with exponential backoff and event deduplication.
    """

    _instance: "EventQueueService | None" = None

    def __new__(cls) -> "EventQueueService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._queue: deque[QueuedEvent] = deque()
        self._processed_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._retry_task: asyncio.Task | None = None
        self._processing = False
        self._initialized = True

        self._persist_file = Path("data/event_queue.json")
        self._persist_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_queue()

        logger.info(
            f"EventQueueService initialized with {len(self._queue)} queued events"
        )

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self._queue if e.status == EventStatus.PENDING)

    def _load_queue(self) -> None:
        """Load queued events from persistent storage."""
        if not self._persist_file.exists():
            return

        try:
            with open(self._persist_file) as f:
                data = json.load(f)
                for item in data.get("queue", []):
                    event = QueuedEvent(**item)
                    self._queue.append(event)
                    self._processed_ids.add(event.id)
            logger.info(f"Loaded {len(self._queue)} events from persistent storage")
        except Exception as e:
            logger.warning(f"Failed to load event queue: {e}")

    def _persist_queue(self) -> None:
        """Persist queued events to storage."""
        try:
            data = {
                "queue": [e.to_dict() for e in self._queue],
                "last_updated": datetime.utcnow().isoformat(),
            }
            with open(self._persist_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist event queue: {e}")

    async def enqueue(
        self,
        event: str,
        data: dict[str, Any],
        room: str | None = None,
        max_retries: int = 5,
    ) -> str:
        """
        Add an event to the queue for later delivery.
        Returns the event ID.
        """
        event_id = str(uuid.uuid4())
        queued_event = QueuedEvent(
            id=event_id,
            event=event,
            data=data,
            room=room,
            created_at=datetime.utcnow().isoformat(),
            max_retries=max_retries,
        )

        async with self._lock:
            self._queue.append(queued_event)
            self._processed_ids.add(event_id)
            self._persist_queue()

        logger.info(f"Enqueued event {event_id}: {event}")
        self._start_retry_worker()
        return event_id

    async def dequeue(self) -> QueuedEvent | None:
        """Remove and return the next event from the queue."""
        async with self._lock:
            if self._queue:
                event = self._queue.popleft()
                self._persist_queue()
                return event
            return None

    async def mark_sent(self, event_id: str) -> None:
        """Mark an event as successfully sent."""
        async with self._lock:
            self._processed_ids.discard(event_id)
            logger.debug(f"Event {event_id} marked as sent")

    async def mark_failed(self, event_id: str, error: str) -> None:
        """Mark an event as failed with error message."""
        async with self._lock:
            for event in self._queue:
                if event.id == event_id:
                    event.retry_count += 1
                    event.last_error = error
                    event.last_retry_at = datetime.utcnow().isoformat()

                    if event.retry_count >= event.max_retries:
                        event.status = EventStatus.EXPIRED
                        logger.warning(
                            f"Event {event_id} expired after {event.retry_count} retries"
                        )
                    else:
                        event.status = EventStatus.FAILED
                        logger.warning(
                            f"Event {event_id} failed (retry {event.retry_count}/{event.max_retries}): {error}"
                        )
                    break

            self._persist_queue()

    def get_pending_events(self) -> list[QueuedEvent]:
        """Get all pending events."""
        return [
            e
            for e in self._queue
            if e.status in (EventStatus.PENDING, EventStatus.FAILED)
        ]

    def get_event_history(self, limit: int = 100) -> list[QueuedEvent]:
        """Get recent event history."""
        return list(self._queue)[:limit]

    def clear_expired(self) -> int:
        """Remove expired events from queue."""
        original_size = len(self._queue)
        self._queue = deque([e for e in self._queue if e.status != EventStatus.EXPIRED])
        cleared = original_size - len(self._queue)
        if cleared > 0:
            self._persist_queue()
            logger.info(f"Cleared {cleared} expired events")
        return cleared

    def _start_retry_worker(self) -> None:
        """Start the background retry worker if not already running."""
        if self._retry_task is None or self._retry_task.done():
            self._retry_task = asyncio.create_task(self._retry_worker())

    async def _retry_worker(self) -> None:
        """Background worker that retries failed events with exponential backoff."""
        if self._processing:
            return

        self._processing = True
        logger.info("Event queue retry worker started")

        try:
            while True:
                await asyncio.sleep(5)

                pending = self.get_pending_events()
                if not pending:
                    continue

                for event in pending:
                    if event.status == EventStatus.EXPIRED:
                        continue

                    backoff = min(2**event.retry_count * 5, 300)
                    last_retry = 0
                    if event.last_retry_at:
                        last_retry = datetime.fromisoformat(
                            event.last_retry_at
                        ).timestamp()
                    time_since_retry = time.time() - last_retry

                    if time_since_retry < backoff:
                        continue

                    logger.info(
                        f"Retrying event {event.id} (attempt {event.retry_count + 1})"
                    )
                    success = await self._retry_event(event)

                    if success:
                        await self.mark_sent(event.id)
                    else:
                        await self.mark_failed(
                            event.id, event.last_error or "Unknown error"
                        )

        except asyncio.CancelledError:
            logger.info("Event queue retry worker cancelled")
        finally:
            self._processing = False

    async def _retry_event(self, event: QueuedEvent) -> bool:
        try:
            from app.api.v1.websockets import emit_to_clients

            sent = await emit_to_clients(event.event, event.data, event.room)
            if sent > 0:
                logger.info("event_delivered_in_process", event_id=event.id, sent=sent)
                return True

            event.last_error = "no_connected_clients"
            return False

        except Exception as e:
            event.last_error = str(e)
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the retry worker."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        self._persist_queue()
        logger.info("EventQueueService shutdown complete")


def get_event_queue() -> EventQueueService:
    return EventQueueService()
