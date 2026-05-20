from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.api.analytics_schemas import AnalyticsQueryResponse
from app.api.routes import _get_chat_pipeline

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WebSocket connected ({len(self.active)} active)")

    async def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)
        logger.info(f"WebSocket disconnected ({len(self.active)} active)")

    async def send_json(self, ws: WebSocket, data: dict[str, Any]) -> None:
        try:
            await ws.send_json(data)
        except Exception:
            await self.disconnect(ws)

    def build_thinking_callback(
        self, ws: WebSocket
    ) -> Callable[[str, str], Coroutine[Any, Any, None]]:
        async def _cb(stage: str, message: str) -> None:
            await self.send_json(
                ws, {"type": "thinking", "stage": stage, "message": message}
            )

        return _cb


manager = ConnectionManager()


@router.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)

            if payload.get("type") != "chat":
                await manager.send_json(
                    ws, {"type": "error", "message": "Expected type='chat'"}
                )
                continue

            question: str = payload.get("question", "")
            messages: list[dict[str, str]] = payload.get("messages", [])
            top_k: int = payload.get("top_k", 5)

            if not question:
                await manager.send_json(
                    ws, {"type": "error", "message": "question is required"}
                )
                continue

            pipeline = _get_chat_pipeline()
            cb = manager.build_thinking_callback(ws)
            response: AnalyticsQueryResponse = await pipeline.run(
                messages, question, top_k, thinking_callback=cb
            )

            await manager.send_json(
                ws,
                {
                    "type": "final",
                    "summary": response.summary,
                    "chart": response.chart.model_dump() if response.chart else None,
                    "sources": [s.model_dump() for s in response.sources],
                    "error": response.error,
                },
            )
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception as exc:
        logger.error(f"WebSocket chat error: {exc}")
        try:
            await manager.send_json(ws, {"type": "error", "message": str(exc)[:500]})
        except Exception:
            pass
        await manager.disconnect(ws)
