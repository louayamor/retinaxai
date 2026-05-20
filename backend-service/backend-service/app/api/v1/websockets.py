import asyncio
import json
from datetime import datetime
from typing import Any
import uuid

import httpx
import redis.asyncio as aioredis
from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import get_settings
from app.db.session import get_db
from app.models.notification import Notification

router = APIRouter()


LLM_SERVICE_URL = "http://localhost:8002"


# Lazy load settings to avoid import errors
def _get_settings():
    try:
        return get_settings()
    except Exception:
        return None


settings = _get_settings()


class EmitRequest(BaseModel):
    event: str
    data: dict[str, Any]
    room: str | None = None


_redis: aioredis.Redis | None = None
_connected_clients: list[WebSocket] = []
_client_rooms: dict[WebSocket, set[str]] = {}


async def emit_to_clients(
    event: str, data: dict[str, Any], room: str | None = None
) -> int:
    message = {"event": event, "data": data}
    target_clients = _get_clients_in_room(room) if room else _connected_clients
    sent_count = 0

    for client in target_clients:
        try:
            await client.send_json(message)
            sent_count += 1
        except Exception as e:
            logger.warning(f"Failed to send bridged websocket event to client: {e}")

    return sent_count


async def get_redis() -> aioredis.Redis | None:
    global _redis
    if _redis is None:
        try:
            redis_url = settings.REDIS_URL if settings else "redis://localhost:6379"
            _redis = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            ping_result = await _redis.ping()
            if ping_result:
                logger.info("WebSocket Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available for WebSocket: {e}")
            _redis = None
    return _redis


def _get_clients_in_room(room: str) -> list[WebSocket]:
    """Get all WebSocket clients subscribed to a specific room."""
    return [client for client, rooms in _client_rooms.items() if room in rooms]


async def emit_prediction_event(
    prediction_id: str,
    patient_id: str,
    status: str,
    dr_grade: int,
    confidence: float,
    overall_severity: str,
    triggers_xai: bool = True,
    error: str | None = None,
) -> None:
    event_type = "prediction.failed" if error else "prediction.completed"
    payload: dict[str, Any] = {
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "status": status,
        "dr_grade": dr_grade,
        "confidence": confidence,
        "overall_severity": overall_severity,
        "triggers_xai": triggers_xai,
        "timestamp": datetime.utcnow().isoformat(),
        "error": error,
    }
    room = f"prediction:{patient_id}"
    await emit_to_clients(event_type, payload, room=room)
    await emit_to_clients(event_type, payload, room=None)

    legacy_payload: dict[str, Any] = {
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "step": "prediction",
        "status": "success",
        "message": f"Prediction complete: DR Grade {dr_grade}, {overall_severity}",
        "timestamp": datetime.utcnow().isoformat(),
    }
    await emit_to_clients("prediction.log", legacy_payload, room=room)
    await emit_to_clients("prediction.log", legacy_payload, room=None)

    logger.info(f"Emitted prediction event: {event_type} for {prediction_id}")


async def emit_xai_event(
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
    payload: dict[str, Any] = {
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
    }
    if patient_id:
        await emit_to_clients(event_type, payload, room=f"prediction:{patient_id}")
    await emit_to_clients(event_type, payload, room=None)

    if patient_id:
        legacy_payload: dict[str, Any] = {
            "prediction_id": prediction_id,
            "patient_id": patient_id,
            "step": "xai",
            "status": "success" if status == "completed" else "info",
            "message": message or f"XAI processing: {event_type}",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await emit_to_clients(
            "prediction.log", legacy_payload, room=f"prediction:{patient_id}"
        )
        await emit_to_clients("prediction.log", legacy_payload, room=None)

    logger.info(f"Emitted XAI event: {event_type} for prediction {prediction_id}")


async def _trigger_llmops_training_workflow(
    job_id: str,
    pipeline: str,
    imaging_version: str | None,
    clinical_version: str | None,
) -> None:
    """Trigger LLMOps workflow after training completes."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/api/workflows/training-complete",
                json={
                    "job_id": job_id,
                    "pipeline": pipeline,
                    "imaging_version": imaging_version,
                    "clinical_version": clinical_version,
                },
            )
            if response.status_code < 400:
                logger.info(f"LLMOps workflow triggered for training {job_id}")
            else:
                logger.warning(
                    f"LLMOps workflow trigger failed: {response.status_code} {response.text}"
                )
    except httpx.ConnectError as e:
        logger.warning(f"LLMOps unavailable for training workflow: {e}")
        await _queue_event_for_retry(
            "llmops.training.workflow",
            {
                "job_id": job_id,
                "pipeline": pipeline,
                "imaging_version": imaging_version,
                "clinical_version": clinical_version,
            },
            None,
        )
    except Exception as e:
        logger.warning(f"Failed to trigger LLMOps workflow: {e}")


async def _queue_event_for_retry(
    event: str,
    data: dict[str, Any],
    room: str | None,
) -> None:
    """Queue an event for later delivery when no clients are connected."""
    try:
        from app.services.event_queue import get_event_queue

        event_queue = get_event_queue()
        await event_queue.enqueue(event, data, room, max_retries=5)
        logger.info(f"Queued event for retry: {event}")
    except Exception as e:
        logger.warning(f"Failed to queue event: {e}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.append(websocket)
    _client_rooms[websocket] = set()
    logger.info(f"Client connected. Total clients: {len(_connected_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")

            try:
                message = json.loads(data) if isinstance(data, str) else data
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"event": "error", "data": {"message": "Invalid JSON"}}
                )
                continue

            event = message.get("event")
            payload = message.get("data", {})

            if event == "subscribe":
                room = payload.get("room")
                if room:
                    _client_rooms[websocket].add(room)
                    await websocket.send_json(
                        {"event": "subscribed", "data": {"room": room}}
                    )
                    logger.info(
                        f"Client subscribed to room: {room}, total rooms: {len(_client_rooms[websocket])}"
                    )

            elif event == "unsubscribe":
                room = payload.get("room")
                if room:
                    _client_rooms[websocket].discard(room)
                    await websocket.send_json(
                        {"event": "unsubscribed", "data": {"room": room}}
                    )

            elif event == "ping":
                await websocket.send_json(
                    {"event": "pong", "data": {"timestamp": payload.get("timestamp")}}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)
        _client_rooms.pop(websocket, None)
        logger.info(f"Client removed. Total clients: {len(_connected_clients)}")


@router.post("/emit")
async def emit_event(request: EmitRequest, db: AsyncSession = Depends(get_db)):
    message = {
        "event": request.event,
        "data": request.data,
    }

    redis = await get_redis()

    target_clients: list[WebSocket] = []

    if request.room:
        target_clients = _get_clients_in_room(request.room)
        logger.info(f"Room {request.room}: targeting {len(target_clients)} clients")

        if redis:
            channel = f"ws:{request.room}"
            await redis.publish(channel, json.dumps(message))
    else:
        target_clients = _connected_clients
        logger.info(f"Broadcast: targeting {len(target_clients)} clients")

        if redis:
            channel = "ws:broadcast"
            await redis.publish(channel, json.dumps(message))

    sent_count = await emit_to_clients(request.event, request.data, request.room)

    logger.debug(f"Emitted {request.event} to {sent_count} clients")

    if sent_count == 0 and len(target_clients) > 0:
        await _queue_event_for_retry(request.event, request.data, request.room)

    # Persist notification to database
    # Also auto-create notifications for training and XAI events
    try:
        notif_data = request.data

        # Direct notification event
        if request.event == "notification":
            notification = Notification(
                id=uuid.UUID(notif_data.get("id", str(uuid.uuid4()))),
                type=notif_data.get("type", "general"),
                title=notif_data.get("title", ""),
                message=notif_data.get("message", ""),
                read=False,
            )
            db.add(notification)
            await db.commit()
            logger.info(f"Persisted notification: {notification.title}")

        # Auto-create notifications for training events
        elif request.event == "training_stage":
            stage = notif_data.get("stage", "")
            status = notif_data.get("status", "")
            message = notif_data.get("message", "")
            pipeline = notif_data.get("pipeline", "unknown")

            # Only create notifications for significant events
            if status in ("completed", "failed") or stage == "pipeline":
                title = f"Training {status.title()}"
                notif_type = "training_error" if status == "failed" else "training"

                notification = Notification(
                    id=uuid.UUID(str(uuid.uuid4())),
                    type=notif_type,
                    title=title,
                    message=f"[{pipeline.upper()}] {message}",
                    read=False,
                )
                db.add(notification)
                await db.commit()
                logger.info(f"Created training notification: {title}")

        # Auto-create notifications for XAI events
        elif request.event and request.event.startswith("xai."):
            stage = notif_data.get("stage", "")
            status = notif_data.get("status", "")
            message = notif_data.get("message", "")

            if status in ("completed", "failed"):
                title = f"XAI {status.title()}"
                notif_type = "error" if status == "failed" else "xai"

                notification = Notification(
                    id=uuid.UUID(str(uuid.uuid4())),
                    type=notif_type,
                    title=title,
                    message=f"[{stage}] {message}",
                    read=False,
                )
                db.add(notification)
                await db.commit()
                logger.info(f"Created XAI notification: {title}")

        # Auto-create notifications for llmops events
        elif request.event in ("llmops_operation", "rag_indexing", "report_generation"):
            status = notif_data.get("status", "")
            message = notif_data.get("message", "")

            if status == "completed":
                title = "LLM Operation Complete"
                notification = Notification(
                    id=uuid.UUID(str(uuid.uuid4())),
                    type="report",
                    title=title,
                    message=message,
                    read=False,
                )
                db.add(notification)
                await db.commit()
                logger.info(f"Created LLM ops notification: {title}")

        # Handle training.completed - trigger LLMOps workflow
        elif request.event == "training.completed":
            job_id = notif_data.get("job_id", "")
            pipeline = notif_data.get("pipeline", "")
            imaging_version = notif_data.get("imaging_version")
            clinical_version = notif_data.get("clinical_version")

            logger.info(
                f"Training completed: job_id={job_id}, pipeline={pipeline}, "
                f"imaging={imaging_version}, clinical={clinical_version}"
            )

            await _trigger_llmops_training_workflow(
                job_id, pipeline, imaging_version, clinical_version
            )

    except Exception as e:
        logger.warning(f"Failed to process notification: {e}")

    return {
        "status": "ok",
        "delivered": sent_count,
        "total_connected": len(_connected_clients),
    }


@router.get("/ws/clients")
async def get_client_count():
    return {"connected": len(_connected_clients)}
