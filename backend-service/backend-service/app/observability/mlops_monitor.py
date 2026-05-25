from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from app.api.v1.websockets import emit_to_clients
from app.services.redis_client import redis_client as shared_redis


DEFAULT_CHANNEL = "mlops.monitor"


def build_monitor_snapshot(
    metrics: dict[str, Any] | None,
    prometheus: dict[str, Any] | None,
    generated_at: str | None,
) -> dict[str, Any]:
    safe_metrics = metrics or {}
    safe_prometheus = prometheus or {} 
    return {
        "generated_at": generated_at,
        "metrics": {
            "imaging": safe_metrics.get("imaging"),
            "clinical": safe_metrics.get("clinical"),
        },
        "training_summary": safe_metrics.get("training_summary"),
        "prometheus": safe_prometheus,
    }


async def _handle_message(payload: str) -> None:
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or "metrics" not in data:
            return
        await emit_to_clients("mlops.monitor", data, room=None)
    except Exception as exc:
        logger.warning("mlops_monitor_message_failed", error=str(exc))


async def subscribe_mlops_monitor(
    channel: str = DEFAULT_CHANNEL,
) -> None:
    backoff = 1
    while True:
        try:
            redis = await shared_redis.get_connection()
            if redis is None:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)
            logger.info("mlops_monitor_subscribed", channel=channel)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                payload = message.get("data")
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                if isinstance(payload, str):
                    await _handle_message(payload)
            backoff = 1
        except Exception as exc:
            logger.warning(
                "mlops_monitor_subscribe_failed",
                error=str(exc),
                backoff_seconds=backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
