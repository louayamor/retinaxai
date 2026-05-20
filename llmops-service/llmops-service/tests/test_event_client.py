from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.event_client import EventClient, send_xai_event


def test_send_xai_event_emits_correct_event_name():
    """send_xai_event emits a fully-qualified event name (e.g. xai.prediction.started)."""
    called = {}

    async def fake_send(url, payload, headers=None, max_retries=2):
        called["url"] = url
        called["body"] = payload
        return True

    with patch("app.services.event_client._send_with_retry", fake_send):
        asyncio.run(
            send_xai_event(
                event="xai.prediction",
                stage="prediction",
                status="started",
                progress=0,
                message="Generating prediction explanation...",
                prediction_id="pred-001",
            )
        )

    assert called["body"]["event"] == "xai.prediction.started"
    assert called["body"]["room"] == "xai:prediction"
    assert called["body"]["data"]["prediction_id"] == "pred-001"
    assert called["body"]["data"]["stage"] == "prediction"
    assert called["body"]["data"]["status"] == "started"


def test_send_xai_event_emits_correct_room_for_each_stage():
    """send_xai_event emits events to the correct xai:<stage> room."""
    calls = []

    async def fake_send(url, payload, headers=None, max_retries=2):
        calls.append(payload)
        return True

    with patch("app.services.event_client._send_with_retry", fake_send):
        for stage in ["prediction", "gradcam", "severity"]:
            asyncio.run(
                send_xai_event(
                    event=f"xai.{stage}",
                    stage=stage,
                    status="completed",
                    progress=100,
                    message=f"{stage} done",
                    prediction_id="pred-002",
                )
            )

    for i, stage in enumerate(["prediction", "gradcam", "severity"]):
        assert calls[i]["room"] == f"xai:{stage}"
        assert calls[i]["event"] == f"xai.{stage}.completed"


def test_send_xai_event_includes_details_and_error():
    """send_xai_event includes optional details and error fields in the payload."""
    called = {}

    async def fake_send(url, payload, headers=None, max_retries=2):
        called["body"] = payload
        return True

    with patch("app.services.event_client._send_with_retry", fake_send):
        asyncio.run(
            send_xai_event(
                event="xai.gradcam",
                stage="gradcam",
                status="failed",
                progress=0,
                message="GradCAM error",
                prediction_id="pred-003",
                details={"left_regions": 3},
                error="model timeout",
            )
        )

    assert called["body"]["event"] == "xai.gradcam.failed"
    assert called["body"]["data"]["error"] == "model timeout"
    assert called["body"]["data"]["details"] == {"left_regions": 3}


def test_send_xai_event_does_not_raise_on_http_error():
    """send_xai_event swallows HTTP errors and does not raise."""

    async def fake_send(url, payload, headers=None, max_retries=2):
        return False

    with patch("app.services.event_client._send_with_retry", fake_send):
        # Should not raise even when sending fails
        asyncio.run(
            send_xai_event(
                event="xai.severity",
                stage="severity",
                status="started",
                progress=0,
                message="Starting severity report",
                prediction_id="pred-004",
            )
        )


def test_send_xai_event_does_not_raise_on_connection_error():
    """send_xai_event swallows connection errors and does not raise."""

    async def fake_send(url, payload, headers=None, max_retries=2):
        return False

    with patch("app.services.event_client._send_with_retry", fake_send):
        asyncio.run(
            send_xai_event(
                event="xai.prediction",
                stage="prediction",
                status="started",
                progress=0,
                message="Starting prediction",
                prediction_id="pred-005",
            )
        )


def test_event_client_is_not_singleton():
    """EventClient instances are independent."""
    a = EventClient()
    b = EventClient()
    assert a is not b


def test_event_client_connect_sets_connected():
    """EventClient.connect sets _connected to True."""
    client = EventClient()
    result = asyncio.run(client.connect())
    assert result is True
    assert client.is_connected is True
    asyncio.run(client.disconnect())


def test_event_client_disconnect_closes_client():
    """EventClient.disconnect closes the underlying httpx client."""
    client = EventClient()
    asyncio.run(client.connect())
    assert client.is_connected is True
    asyncio.run(client.disconnect())
    assert client.is_connected is False
