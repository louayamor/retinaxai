from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import Notification

logger = structlog.get_logger(__name__)


class NotificationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_notification(
        self,
        notif_type: str,
        title: str,
        message: str,
        user_id: uuid.UUID | None = None,
    ) -> Notification:
        notification = Notification(
            id=uuid.uuid4(),
            type=notif_type,
            title=title,
            message=message,
            read=False,
            user_id=user_id,
        )
        self.db.add(notification)
        await self.db.commit()
        logger.info("notification_created", type=notif_type, title=title)
        return notification

    async def process_event_notification(
        self, event: str, data: dict[str, Any]
    ) -> None:
        if event == "notification":
            await self.create_notification(
                notif_type=data.get("type", "general"),
                title=data.get("title", ""),
                message=data.get("message", ""),
            )
        elif event == "training_stage":
            await self._process_training_event(data)
        elif event and event.startswith("xai."):
            await self._process_xai_event(data)
        elif event in ("llmops_operation", "rag_indexing", "report_generation"):
            await self._process_llmops_event(event, data)

    async def _process_training_event(self, data: dict[str, Any]) -> None:
        status = data.get("status", "")
        if status not in ("completed", "failed"):
            return
        title = f"Training {status.title()}"
        await self.create_notification(
            notif_type="training_error" if status == "failed" else "training",
            title=title,
            message=f"[{data.get('pipeline', 'unknown').upper()}] {data.get('message', '')}",
        )

    async def _process_xai_event(self, data: dict[str, Any]) -> None:
        status = data.get("status", "")
        if status not in ("completed", "failed"):
            return
        title = f"XAI {status.title()}"
        await self.create_notification(
            notif_type="error" if status == "failed" else "xai",
            title=title,
            message=f"[{data.get('stage', '')}] {data.get('message', '')}",
        )

    async def _process_llmops_event(self, event: str, data: dict[str, Any]) -> None:
        status = data.get("status", "")
        if status == "completed":
            await self.create_notification(
                notif_type="report",
                title="LLM Operation Complete",
                message=data.get("message", ""),
            )
