from __future__ import annotations
import asyncio
import json
from datetime import datetime
from typing import Any

import httpx
import websockets.exceptions
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.db.session import get_db
from app.notifications.service import NotificationService
import redis.asyncio as aioredis
from app.services.redis_client import RedisClient, redis_client as shared_redis

router = APIRouter()

WS_CHAT_RECEIVE_TIMEOUT: float = 120.0





class EmitRequest(BaseModel):
    event: str
    data: dict[str, Any]
    room: str | None = None


class _WebSocketManager:
    def __init__(self) -> None:
        self.connected_clients: list[WebSocket] = []
        self.client_rooms: dict[WebSocket, set[str]] = {}


_ws_manager = _WebSocketManager()


async def emit_to_clients(
    event: str, data: dict[str, Any], room: str | None = None
) -> int:
    message = {"event": event, "data": data}
    target_clients = _get_clients_in_room(room) if room else _ws_manager.connected_clients
    sent_count = 0

    for client in target_clients:
        try:
            await client.send_json(message)
            sent_count += 1
        except Exception as e:
            logger.warning(f"Failed to send bridged websocket event to client: {e}")

    return sent_count




def _get_clients_in_room(room: str) -> list[WebSocket]:
    """Get all WebSocket clients subscribed to a specific room."""
    return [client for client, rooms in _ws_manager.client_rooms.items() if room in rooms]


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
    headers = {"Content-Type": "application/json"}
    if settings.LLM_SERVICE_API_KEY:
        headers["X-API-Key"] = settings.LLM_SERVICE_API_KEY

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.LLM_SERVICE_URL}/api/workflows/training-complete",
                json={
                    "job_id": job_id,
                    "pipeline": pipeline,
                    "imaging_version": imaging_version,
                    "clinical_version": clinical_version,
                },
                headers=headers,
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


async def _listen_ws_broadcast() -> None:
    backoff = 1
    patterns = ["ws:*"]
    redis: aioredis.Redis | None = None
    while True:
        try:
            redis = await RedisClient.create_dedicated_connection()
            if redis is None:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15)
                continue

            pubsub = redis.pubsub()
            for pattern in patterns:
                await pubsub.psubscribe(pattern)
            logger.info("ws_broadcast_listener_started", patterns=patterns)

            async for message in pubsub.listen():
                if message.get("type") not in ("pmessage", "message"):
                    continue
                channel: str = message.get("channel", "") or ""
                data_raw = message.get("data")
                if isinstance(data_raw, bytes):
                    data_raw = data_raw.decode("utf-8")
                if not isinstance(data_raw, str):
                    continue

                try:
                    payload = json.loads(data_raw)
                except json.JSONDecodeError:
                    continue

                event = payload.get("event", "")
                event_data = payload.get("data", {})

                room: str | None = None
                if channel.startswith("ws:") and channel != "ws:broadcast":
                    room = channel.removeprefix("ws:")

                if room or event_data:
                    await emit_to_clients(event, event_data, room=room)

        except Exception as exc:
            logger.warning("ws_broadcast_listener_error", error=str(exc), exc_type=type(exc).__name__)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)
        finally:
            if redis is not None:
                try:
                    await redis.close()
                except Exception:
                    pass


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
    _ws_manager.connected_clients.append(websocket)
    _ws_manager.client_rooms[websocket] = set()
    logger.info(f"Client connected. Total clients: {len(_ws_manager.connected_clients)}")

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
                    _ws_manager.client_rooms[websocket].add(room)
                    await websocket.send_json(
                        {"event": "subscribed", "data": {"room": room}}
                    )
                    logger.info(
                        f"Client subscribed to room: {room}, total rooms: {len(_ws_manager.client_rooms[websocket])}"
                    )

            elif event == "unsubscribe":
                room = payload.get("room")
                if room:
                    _ws_manager.client_rooms[websocket].discard(room)
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
        if websocket in _ws_manager.connected_clients:
            _ws_manager.connected_clients.remove(websocket)
        _ws_manager.client_rooms.pop(websocket, None)
        logger.info(f"Client removed. Total clients: {len(_ws_manager.connected_clients)}")


@router.websocket("/ws/chat")
async def chat_websocket_proxy(websocket: WebSocket):
    await websocket.accept()

    token = websocket.query_params.get("token", "")
    target_url = f"{settings.LLM_SERVICE_URL}/ws/chat"
    if token:
        target_url += f"?token={token}"

    try:
        async with websockets.connect(target_url) as llm_ws:
            async def forward_to_client() -> None:
                try:
                    async for msg in llm_ws:
                        await websocket.send_text(msg if isinstance(msg, str) else json.dumps(msg))
                except Exception:
                    pass

            forward_task = asyncio.create_task(forward_to_client())

            try:
                while True:
                    data = await websocket.receive_text()
                    await llm_ws.send(data)
            except WebSocketDisconnect:
                pass
            finally:
                forward_task.cancel()
                try:
                    await forward_task
                except (asyncio.CancelledError, Exception):
                    pass
    except (websockets.exceptions.WebSocketException, OSError) as e:
        logger.error("chat_ws_proxy_failed", error=str(e))
        await websocket.send_json({"event": "error", "data": {"message": "Chat service unavailable"}})


@router.post("/emit")
async def emit_event(request: EmitRequest, db: AsyncSession = Depends(get_db)):
    message = {
        "event": request.event,
        "data": request.data,
    }

    redis = await shared_redis.get_connection()

    target_clients: list[WebSocket] = []

    if request.room:
        target_clients = _get_clients_in_room(request.room)
        logger.info(f"Room {request.room}: targeting {len(target_clients)} clients")

        if redis:
            channel = f"ws:{request.room}"
            await redis.publish(channel, json.dumps(message))
    else:
        target_clients = _ws_manager.connected_clients
        logger.info(f"Broadcast: targeting {len(target_clients)} clients")

        if redis:
            channel = "ws:broadcast"
            await redis.publish(channel, json.dumps(message))

    sent_count = await emit_to_clients(request.event, request.data, request.room)

    logger.debug(f"Emitted {request.event} to {sent_count} clients")

    if sent_count == 0 and len(target_clients) > 0:
        await _queue_event_for_retry(request.event, request.data, request.room)

    try:
        notif_service = NotificationService(db)
        await notif_service.process_event_notification(request.event, request.data)

        # Handle training.completed - trigger LLMOps workflow
        if request.event == "training.completed":
            notif_data = request.data
            await _trigger_llmops_training_workflow(
                job_id=notif_data.get("job_id", ""),
                pipeline=notif_data.get("pipeline", ""),
                imaging_version=notif_data.get("imaging_version"),
                clinical_version=notif_data.get("clinical_version"),
            )
    except Exception as e:
        logger.warning(f"Failed to process notification: {e}")

    return {
        "status": "ok",
        "delivered": sent_count,
        "total_connected": len(_ws_manager.connected_clients),
    }


@router.get("/ws/clients")
async def get_client_count():
    return {"connected": len(_ws_manager.connected_clients)}
