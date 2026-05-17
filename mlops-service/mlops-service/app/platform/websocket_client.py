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


class WebSocketClient:
    _instance: "WebSocketClient | None" = None

    def __init__(self) -> None:
        self._base_url = BACKEND_WS_URL.replace("ws://", "http://").replace("/ws", "")

    @classmethod
    def get_instance(cls) -> "WebSocketClient":
        if cls._instance is None:
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

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    emit_url,
                    json={
                        "event": "training_stage",
                        "data": payload.get("data", {}),
                        "room": room,
                    },
                    headers={"X-API-Key": BACKEND_API_KEY} if BACKEND_API_KEY else {},
                )
                if response.status_code < 400:
                    logger.debug(
                        f"Sent training event: {stage} - {status} to room {room}"
                    )
                else:
                    logger.warning(
                        f"Failed to emit event: {response.status_code} {response.text}"
                    )
        except Exception as e:
            logger.warning(f"Failed to send training event: {e}")


@lru_cache(maxsize=1)
def get_websocket_client() -> WebSocketClient:
    """Get WebSocket client instance (cached singleton)."""
    return WebSocketClient.get_instance()


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

    sanitized_payload = _sanitize_json(payload)
    json_data = json.dumps(
        {
            "event": event_type,
            "data": sanitized_payload,
            "room": room,
        },
        allow_nan=False,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                emit_url,
                content=json_data,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": BACKEND_API_KEY,
                }
                if BACKEND_API_KEY
                else {"Content-Type": "application/json"},
            )
            if response.status_code < 400:
                logger.debug(f"Sent prediction event: {event_type} for {prediction_id}")
            else:
                logger.warning(
                    f"Failed to emit prediction event: {response.status_code}"
                )
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed (NaN/Inf in payload): {e}")
        logger.debug(f"Payload keys: {list(payload.keys())}")
    except Exception as e:
        logger.warning(f"Failed to send prediction event: {e}")


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

    sanitized_payload = _sanitize_json(payload)
    json_data = json.dumps(
        {
            "event": "prediction.log",
            "data": sanitized_payload,
            "room": room,
        },
        allow_nan=False,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                emit_url,
                content=json_data,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": BACKEND_API_KEY,
                }
                if BACKEND_API_KEY
                else {"Content-Type": "application/json"},
            )
            if response.status_code < 400:
                logger.debug(f"Sent log: {step} - {status}")
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed in send_prediction_log: {e}")
    except Exception as e:
        logger.warning(f"Failed to send prediction log: {e}")
