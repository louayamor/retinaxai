from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession


class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(self, user_id: uuid.UUID, title: str = "New Chat") -> ChatSession:
        session = ChatSession(user_id=user_id, title=title)
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session(self, session_id: str) -> ChatSession | None:
        return await self.db.get(ChatSession, session_id)

    async def list_user_sessions(
        self, user_id: uuid.UUID
    ) -> list[tuple[ChatSession, int]]:
        result = await self.db.execute(
            sa.select(ChatSession, sa.func.count(ChatMessage.id).label("message_count"))
            .outerjoin(ChatMessage)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.updated_at.desc())
        )
        return result.all()

    async def update_session(self, session: ChatSession) -> None:
        await self.db.commit()

    async def delete_session(self, session: ChatSession) -> None:
        await self.db.delete(session)
        await self.db.commit()

    async def add_message(self, message: ChatMessage) -> ChatMessage:
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        result = await self.db.execute(
            sa.select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return list(result.scalars().all())
