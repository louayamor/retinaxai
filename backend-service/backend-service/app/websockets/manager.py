import os
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from app.core.config import get_settings
from loguru import logger
import socketio
from socketio import AsyncManager, AsyncServer, AsyncNamespace


class WebSocketNamespace(AsyncNamespace):
    async def on_connect(self, sid: str, environ: dict[str, Any]) -> None:
        logger.info(f"Client connected: sid={sid}")
        await self.emit(
            "connected", {"sid": sid, "message": "Connected to RetinaXAI WebSocket"}
        )

    async def on_disconnect(self, sid: str) -> None:
        logger.info(f"Client disconnected: sid={sid}")

    async def on_subscribe(self, sid: str, data: dict[str, Any]) -> None:
        room = data.get("room")
        if room:
            await self.enter_room(sid, room)
            logger.info(f"Client {sid} subscribed to room: {room}")

    async def on_unsubscribe(self, sid: str, data: dict[str, Any]) -> None:
        room = data.get("room")
        if room:
            await self.leave_room(sid, room)
            logger.info(f"Client {sid} unsubscribed from room: {room}")


class SocketManager:
    _instance: "SocketManager | None" = None
    _server: AsyncServer | None = None
    _redis: aioredis.Redis | None = None
    _namespace: WebSocketNamespace

    def __init__(self) -> None:
        self.settings = get_settings()
        self._namespace = WebSocketNamespace("/")

    @classmethod
    def get_instance(cls) -> "SocketManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        redis_url = self.settings.REDIS_URL
        logger.info(f"Initializing SocketManager with Redis: {redis_url}")

        try:
            self._redis = aioredis.from_url(
                redis_url, encoding="utf-8", decode_responses=True
            )
            ping_result = await self._redis.ping()
            if ping_result:
                logger.info("Redis connection verified")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to in-memory: {e}")
            self._redis = None

        self._server = AsyncServer(
            async_mode="asyncio",
            cors_allowed_origins="*",
            manager=AsyncManager() if not self._redis else None,
        )
        if self._server is not None:
            self._server.on("connect", self._namespace.on_connect)
            self._server.on("disconnect", self._namespace.on_disconnect)
            self._server.on("subscribe", self._namespace.on_subscribe)
            self._server.on("unsubscribe", self._namespace.on_unsubscribe)

        logger.info("SocketManager initialized")

    @property
    def server(self) -> AsyncServer:
        if self._server is None:
            raise RuntimeError(
                "SocketManager not initialized. Call initialize() first."
            )
        return self._server

    async def emit_to_room(self, room: str, event: str, data: dict[str, Any]) -> None:
        if self._server:
            await self._server.emit(event, data, room=room)
            logger.debug(f"Emitted {event} to room {room}: {data}")

    async def emit_to_all(self, event: str, data: dict[str, Any]) -> None:
        if self._server:
            await self._server.emit(event, data)
            logger.debug(f"Emitted {event} to all: {data}")

    async def emit_training_event(
        self,
        job_id: str,
        pipeline: str,
        stage: str,
        status: str,
        progress: int,
        message: str | None = None,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "event": "training_stage",
            "data": {
                "job_id": job_id,
                "pipeline": pipeline,
                "stage": stage,
                "status": status,
                "progress": progress,
                "message": message,
                "metrics": metrics,
                "error": error,
            },
        }
        room = f"training:{pipeline}"
        await self.emit_to_room(room, "training_stage", payload)
        await self.emit_to_room(f"job:{job_id}", "training_stage", payload)
        logger.info(f"Emitted training event: {stage} - {status} ({progress}%)")

    async def emit_llmops_event(
        self,
        event_type: str,
        status: str,
        progress: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event": event_type,
            "data": {
                "status": status,
                "progress": progress,
                "message": message,
                "details": details or {},
            },
        }
        await self.emit_to_all(event_type, payload)
        logger.info(f"Emitted LLM ops event: {event_type} - {status}")

    async def emit_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
        user_id: str | None = None,
    ) -> None:
        """Emit a notification event and also broadcast to notification room."""
        payload = {
            "event": "notification",
            "data": {
                "id": str(uuid.uuid4()),
                "type": notification_type,
                "title": title,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "read": False,
                "user_id": user_id,
            },
        }
        await self.emit_to_all("notification", payload)
        logger.info(f"Emitted notification: {notification_type} - {title}")

    async def emit_prediction_event(
        self,
        prediction_id: str,
        patient_id: str,
        status: str,
        dr_grade: int,
        confidence: float,
        overall_severity: str,
        triggers_xai: bool = True,
        error: str | None = None,
    ) -> None:
        """Emit prediction.completed or prediction.failed event."""
        event_type = "prediction.failed" if error else "prediction.completed"
        payload = {
            "event": event_type,
            "data": {
                "prediction_id": prediction_id,
                "patient_id": patient_id,
                "status": status,
                "dr_grade": dr_grade,
                "confidence": confidence,
                "overall_severity": overall_severity,
                "triggers_xai": triggers_xai,
                "timestamp": datetime.utcnow().isoformat(),
                "error": error,
            },
        }
        room = f"prediction:{patient_id}"
        await self.emit_to_room(room, event_type, payload)
        await self.emit_to_all(event_type, payload)

        legacy_payload = {
            "event": "prediction.log",
            "data": {
                "prediction_id": prediction_id,
                "patient_id": patient_id,
                "step": "prediction",
                "status": "success",
                "message": f"Prediction complete: DR Grade {dr_grade}, {overall_severity}",
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        await self.emit_to_room(room, "prediction.log", legacy_payload)
        await self.emit_to_all("prediction.log", legacy_payload)

        logger.info(f"Emitted prediction event: {event_type} for {prediction_id}")

    async def emit_xai_event(
        self,
        event_type: str,
        prediction_id: str,
        patient_id: str | None,
        status: str,
        progress: int,
        message: str,
        explanation_id: str | None = None,
        content: str | None = None,
        summary: str | None = None,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Emit XAI events (xai.explanation_ready, xai.gradcam_ready, xai.severity_ready)."""
        payload = {
            "event": event_type,
            "data": {
                "prediction_id": prediction_id,
                "patient_id": patient_id,
                "status": status,
                "progress": progress,
                "message": message,
                "explanation_id": explanation_id,
                "content": content,
                "summary": summary,
                "details": details or {},
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        if patient_id:
            await self.emit_to_room(f"prediction:{patient_id}", event_type, payload)
        await self.emit_to_all(event_type, payload)

        if patient_id:
            legacy_xai_payload = {
                "event": "prediction.log",
                "data": {
                    "prediction_id": prediction_id,
                    "patient_id": patient_id,
                    "step": "xai",
                    "status": "success" if status == "completed" else "info",
                    "message": message or f"XAI processing: {event_type}",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
            await self.emit_to_room(f"prediction:{patient_id}", "prediction.log", legacy_xai_payload)
            await self.emit_to_all("prediction.log", legacy_xai_payload)

        logger.info(f"Emitted XAI event: {event_type} for prediction {prediction_id}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed")
        self._server = None
        SocketManager._instance = None


def get_socket_manager() -> SocketManager:
    return SocketManager.get_instance()
