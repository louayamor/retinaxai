"""
Operation state tracker for LLMOps service.
Tracks current operation: indexing, retrieval, generation.
"""

import asyncio
from threading import Lock
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from loguru import logger

_operation_lock = Lock()
_current_operation: dict | None = None

_emit_history: list[dict] = []


def _try_emit_ws(
    event_type: str,
    status: str,
    progress: int,
    message: str,
    details: dict | None = None,
) -> None:
    """Try to emit WebSocket event without blocking caller flow."""
    try:
        from app.services.websocket_client import get_websocket_client

        ws_client = get_websocket_client()
        coro = ws_client.send_llmops_event(
            event_type, status, progress, message, details
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)
    except Exception as exc:
        logger.warning(f"Failed to emit llmops operation event: {exc}")


@dataclass
class OperationState:
    state: Literal["idle", "indexing", "retrieving", "generating"]
    message: str
    progress: float | None = None
    started_at: str | None = None


def set_operation(state: str, message: str, progress: float | None = None):
    global _current_operation
    with _operation_lock:
        _current_operation = {
            "state": state,
            "message": message,
            "progress": progress,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    status_map = {
        "idle": "completed",
        "indexing": "running",
        "retrieving": "running",
        "generating": "running",
    }
    _try_emit_ws(
        event_type="llmops_operation",
        status=status_map.get(state, "running"),
        progress=int(progress or 0),
        message=message,
    )


def clear_operation():
    global _current_operation
    with _operation_lock:
        _current_operation = None
    _try_emit_ws(
        event_type="llmops_operation",
        status="completed",
        progress=100,
        message="Operation cleared",
    )


def get_operation() -> OperationState:
    with _operation_lock:
        if _current_operation is None:
            return OperationState(state="idle", message="Ready")
        return OperationState(**_current_operation)


class OperationStateManager:
    """Wrapper exposing module-level state operations as methods."""

    def set_operation(
        self, state: str, message: str, progress: float | None = None
    ) -> None:
        set_operation(state, message, progress)

    def get_operation(self) -> OperationState:
        return get_operation()


_operation_manager: OperationStateManager | None = None


def get_operation_state_manager() -> OperationStateManager:
    """FastAPI dependency factory. Creates instance if not overridden."""
    global _operation_manager
    if _operation_manager is None:
        _operation_manager = OperationStateManager()
    return _operation_manager
