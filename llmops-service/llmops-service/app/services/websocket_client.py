import asyncio
import os
from typing import Any

import httpx
from loguru import logger

BACKEND_WS_URL = os.environ.get("BACKEND_WS_URL", "http://localhost:8000")
LLMOPS_API_KEY = os.environ.get("LLM_SERVICE_API_KEY", "")
MAX_RETRIES = 2
INITIAL_BACKOFF = 0.5


class WebSocketClient:
    _instance: "WebSocketClient | None" = None
    _connected: bool = False

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._url = BACKEND_WS_URL

    @classmethod
    def get_instance(cls) -> "WebSocketClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self) -> bool:
        if self._connected:
            return True

        try:
            self._client = httpx.AsyncClient(timeout=5.0)
            self._connected = True
            logger.info("LLMOps WebSocket client initialized")
            return True
        except Exception as e:
            logger.warning(f"WebSocket client init failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _send_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> bool:
        """Send HTTP request with exponential backoff retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url, json=payload, headers=headers or {}
                    )
                    if response.status_code < 400:
                        return True
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {last_error}"
                    )
            except httpx.ConnectError as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                backoff = INITIAL_BACKOFF * (2**attempt)
                logger.info(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        return False

    async def send_llmops_event(
        self,
        event_type: str,
        status: str,
        progress: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self._connected:
            await self.connect()

        payload = {
            "event": event_type,
            "data": {
                "status": status,
                "progress": progress,
                "message": message,
                "details": details or {},
            },
        }

        room = "llmops" if event_type in ("rag_indexing", "report_generation") else None

        emit_payload = {"event": event_type, "data": payload.get("data", {})}
        if room:
            emit_payload["room"] = room

        headers = {"X-API-Key": LLMOPS_API_KEY} if LLMOPS_API_KEY else {}
        url = self._url.replace("ws", "http") + "/emit"

        success = await self._send_with_retry(url, emit_payload, headers)

        if success:
            logger.debug(
                f"Sent LLM ops event: {event_type} - {status} to room {room or 'broadcast'}"
            )
        else:
            logger.error(
                f"Failed to send LLM ops event after {MAX_RETRIES} retries: {event_type}"
            )


_websocket_client: WebSocketClient | None = None


def get_websocket_client() -> WebSocketClient:
    global _websocket_client
    if _websocket_client is None:
        _websocket_client = WebSocketClient.get_instance()
    return _websocket_client


async def send_xai_event(
    event: str,
    stage: str,
    status: str,
    progress: int,
    message: str,
    prediction_id: str,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Send XAI event to backend WebSocket server with retry."""
    payload = {
        "event": event,
        "data": {
            "prediction_id": prediction_id,
            "stage": stage,
            "status": status,
            "progress": progress,
            "message": message,
            "details": details or {},
            "error": error,
        },
    }

    room = f"xai:{stage}"
    emit_url = f"{BACKEND_WS_URL}/emit"

    emit_payload = {
        "event": f"{event}.{status}",
        "data": payload.get("data", {}),
        "room": room,
    }

    headers = {"X-API-Key": LLMOPS_API_KEY} if LLMOPS_API_KEY else {}

    logger.debug(f"Sending XAI event {stage} - {status} to {emit_url}")
    success = await _send_with_retry(emit_url, emit_payload, headers)

    if success:
        logger.debug(f"Sent XAI event: {stage} - {status}")
    else:
        logger.warning(f"Failed to send XAI event after {MAX_RETRIES} retries: {event} - continuing anyway")


async def _send_with_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """Send HTTP request with exponential backoff retry."""
    last_error = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.post(url, json=payload, headers=headers or {})
                if response.status_code < 400:
                    return True
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {last_error}"
                )
        except httpx.ConnectError as e:
            last_error = str(e)
            logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        except Exception as e:
            last_error = str(e)
            logger.debug(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

        if attempt < max_retries - 1:
            backoff = INITIAL_BACKOFF * (2**attempt)
            logger.debug(f"Retrying in {backoff}s...")
            await asyncio.sleep(backoff)

    logger.debug(f"All {max_retries} attempts failed. Last error: {last_error}")
    return False
