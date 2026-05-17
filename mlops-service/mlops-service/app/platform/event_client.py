from __future__ import annotations

import asyncio
import json
import math
import os
from functools import lru_cache
from typing import Any

import httpx
from loguru import logger

BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "ws://localhost:8000/ws")
BACKEND_API_KEY = os.environ.get("ML_SERVICE_API_KEY", "")
MAX_RETRIES = 2
INITIAL_BACKOFF = 0.5


def _sanitize_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


class EventClient:
    def __init__(self) -> None:
        self._base_url = BACKEND_WS_URL.replace("ws://", "http://").replace("/ws", "")

    @classmethod
    def get_instance(cls) -> EventClient:
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=10.0)

    async def send_training_event(
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
        emit_url = f"{self._base_url}/emit"
        headers = {"X-API-Key": BACKEND_API_KEY} if BACKEND_API_KEY else {}

        success = await _send_with_retry(
            emit_url,
            {"event": "training_stage", "data": payload.get("data", {}), "room": room},
            headers,
        )
        if success:
            logger.debug(f"Sent training event: {stage} - {status} to room {room}")


async def _send_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_retries: int = MAX_RETRIES,
) -> bool:
    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=headers or {})
                if response.status_code < 400:
                    return True
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {last_error}"
                )
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            backoff = INITIAL_BACKOFF * (2**attempt)
            await asyncio.sleep(backoff)
    logger.warning(f"All {max_retries} attempts failed. Last error: {last_error}")
    return False


@lru_cache(maxsize=1)
def get_event_client() -> EventClient:
    """Get event client instance (cached singleton)."""
    return EventClient.get_instance()


async def send_prediction_event(
    prediction_id: str,
    patient_id: str,
    dr_grade: int,
    confidence: float,
    imaging_confidence: float,
    clinical_confidence: float | None,
    combined_grade: int,
    overall_severity: str,
    triggers_xai: bool = True,
    error: str | None = None,
) -> None:
    """Send prediction.completed event to backend WebSocket server."""
    from datetime import datetime

    # Pre-sanitize float parameters before building payload
    if math.isnan(confidence) or math.isinf(confidence):
        confidence = 0.0
    if math.isnan(imaging_confidence) or math.isinf(imaging_confidence):
        imaging_confidence = 0.0
    if clinical_confidence is not None and (
        math.isnan(clinical_confidence) or math.isinf(clinical_confidence)
    ):
        clinical_confidence = None

    event_type = "prediction.failed" if error else "prediction.completed"
    payload = {
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "dr_grade": dr_grade,
        "confidence": confidence,
        "imaging_confidence": imaging_confidence,
        "clinical_confidence": clinical_confidence,
        "combined_grade": combined_grade,
        "overall_severity": overall_severity,
        "triggers_xai": triggers_xai,
        "timestamp": datetime.utcnow().isoformat(),
        "error": error,
    }

    room = f"prediction:{patient_id}"
    emit_url = f"{BACKEND_WS_URL.replace('ws://', 'http://').replace('/ws', '')}/emit"

    try:
        sanitized_payload = _sanitize_json(payload)
        json_data = json.dumps(
            {"event": event_type, "data": sanitized_payload, "room": room},
            allow_nan=False,
        )
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed (NaN/Inf in payload): {e}")
        logger.debug(f"Payload keys: {list(payload.keys())}")
        return

    headers: dict[str, str] = (
        {"Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY}
        if BACKEND_API_KEY
        else {"Content-Type": "application/json"}
    )

    success = await _send_with_retry(emit_url, json.loads(json_data), headers)
    if success:
        logger.debug(f"Sent prediction event: {event_type} for {prediction_id}")


async def send_raw_event(
    event: str,
    data: dict[str, Any],
    room: str = "notifications",
) -> bool:
    """Send a generic event to the backend WebSocket server.

    Args:
        event: Event name (e.g. "notification", "model.registered").
        data: Payload to send.
        room: WebSocket room to broadcast to.

    Returns:
        True if the event was sent successfully.
    """
    from datetime import datetime

    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat()

    emit_url = f"{BACKEND_WS_URL.replace('ws://', 'http://').replace('/ws', '')}/emit"

    headers: dict[str, str] = (
        {"Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY}
        if BACKEND_API_KEY
        else {"Content-Type": "application/json"}
    )

    payload = {
        "event": event,
        "data": _sanitize_json(data),
        "room": room,
    }

    success = await _send_with_retry(emit_url, payload, headers)
    if success:
        logger.debug(f"Sent event: {event} to room {room}")
    return success


async def send_notification(
    title: str,
    message: str,
    notif_type: str = "general",
    room: str = "notifications",
) -> bool:
    """Send a notification event to the backend for persistence + broadcast."""
    return await send_raw_event(
        event="notification",
        data={
            "id": "",
            "type": notif_type,
            "title": title,
            "message": message,
        },
        room=room,
    )


async def send_prediction_log(
    patient_id: str,
    prediction_id: str,
    step: str,
    status: str,
    message: str,
) -> None:
    """Send live prediction log message to backend WebSocket server for frontend display."""
    from datetime import datetime

    payload = {
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "step": step,
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    room = f"prediction:{patient_id}"
    emit_url = f"{BACKEND_WS_URL.replace('ws://', 'http://').replace('/ws', '')}/emit"

    try:
        sanitized_payload = _sanitize_json(payload)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed in send_prediction_log: {e}")
        return

    headers: dict[str, str] = (
        {"Content-Type": "application/json", "X-API-Key": BACKEND_API_KEY}
        if BACKEND_API_KEY
        else {"Content-Type": "application/json"}
    )

    emit_payload = {
        "event": "prediction.log",
        "data": sanitized_payload,
        "room": room,
    }

    success = await _send_with_retry(emit_url, emit_payload, headers)
    if success:
        logger.debug(f"Sent log: {step} - {status}")
