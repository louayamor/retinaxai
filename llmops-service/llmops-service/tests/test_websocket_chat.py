from __future__ import annotations

import pytest

from fastapi.testclient import TestClient


pytestmark = pytest.mark.skip(reason="requires llmops-service dev dependencies")


def test_ws_chat_returns_error_when_pipeline_raises(monkeypatch) -> None:
    """WS should respond with a structured error, not crash the connection."""
    from app.main import create_app

    app = create_app()

    class DummyPipeline:
        async def run(self, messages, question, top_k, thinking_callback=None):
            raise RuntimeError("boom")

    import app.api.ws_handlers as ws_handlers

    monkeypatch.setattr(ws_handlers, "_get_chat_pipeline", lambda: DummyPipeline())

    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "chat", "question": "hi", "messages": [], "top_k": 1})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "boom" in msg["message"]


def test_ws_chat_accepts_multiple_messages(monkeypatch) -> None:
    from app.main import create_app

    app = create_app()

    class DummyResp:
        summary = "ok"
        chart = None
        sources = []
        error = None

    class DummyPipeline:
        async def run(self, messages, question, top_k, thinking_callback=None):
            return DummyResp()

    import app.api.ws_handlers as ws_handlers

    monkeypatch.setattr(ws_handlers, "_get_chat_pipeline", lambda: DummyPipeline())

    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "chat", "question": "q1", "messages": []})
        msg1 = ws.receive_json()
        assert msg1["type"] == "final"
        assert msg1["summary"] == "ok"

        ws.send_json({"type": "chat", "question": "q2", "messages": []})
        msg2 = ws.receive_json()
        assert msg2["type"] == "final"
        assert msg2["summary"] == "ok"
