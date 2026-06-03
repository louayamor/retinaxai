from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


from app.chat.service import ChatService


class DummyDB:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False

    async def get(self, model, pk: str) -> object | None:
        if pk == "session-1":
            return SimpleNamespace(
                id="session-1",
                user_id="user-1",
                title="New Chat",
                updated_at=None,
                created_at=None,
            )
        return SimpleNamespace(
            id="session-2",
            user_id="user-2",
            title="Other",
        )

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True

    async def flush(self) -> None:
        pass

    async def refresh(self, obj: object) -> None:
        if hasattr(obj, "id") and not obj.id:
            obj.id = "new-id"

    async def execute(self, stmt: object) -> SimpleNamespace:
        return SimpleNamespace(
            all=lambda: [],
            scalars=lambda: SimpleNamespace(all=lambda: []),
        )

    def delete(self, obj: object) -> None:
        pass


@pytest.mark.asyncio
async def test_chat_service_create_session() -> None:
    db = DummyDB()
    service = ChatService(db)
    result = await service.create_session("user-1")
    assert result.session_id == "new-id"
    assert result.title == "New Chat"
    assert db.committed


@pytest.mark.asyncio
async def test_chat_service_get_session_not_found() -> None:
    db = DummyDB()
    service = ChatService(db)
    with pytest.raises(HTTPException):
        await service.get_session("nonexistent", "user-1")


@pytest.mark.asyncio
async def test_chat_service_update_session_title() -> None:
    db = DummyDB()
    service = ChatService(db)
    result = await service.update_session_title("session-1", "user-1", "New Title")
    assert result.title == "New Title"


@pytest.mark.asyncio
async def test_chat_service_delete_session_wrong_user() -> None:
    db = DummyDB()
    service = ChatService(db)
    with pytest.raises(HTTPException):
        await service.delete_session("session-2", "user-1")


@pytest.mark.asyncio
async def test_chat_service_send_message_missing_session() -> None:
    db = DummyDB()
    service = ChatService(db)
    with pytest.raises(HTTPException):
        await service.send_message("nonexistent", "user-1", "hello")
