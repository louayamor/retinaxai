from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
import structlog
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatRole, ChatSession
from app.schemas.chat_schemas import (
    ChatMessageSchema,
    ChatSessionDetailSchema,
    ChatSessionSchema,
    CreateChatSessionResponse,
    SendMessageResponse,
)
from app.services.llm_client.chat_client import ChatServiceClient, chat_client

logger = structlog.get_logger(__name__)


class ChatService:
    def __init__(
        self,
        db: AsyncSession,
        chat_client_override: ChatServiceClient | None = None,
    ):
        self.db = db
        self._chat_client = chat_client_override or chat_client

    async def create_session(self, user_id: uuid.UUID) -> CreateChatSessionResponse:
        session = ChatSession(user_id=user_id, title="New Chat")
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return CreateChatSessionResponse(session_id=str(session.id), title=session.title)

    async def list_sessions(self, user_id: uuid.UUID) -> list[ChatSessionSchema]:
        result = await self.db.execute(
            sa.select(ChatSession, sa.func.count(ChatMessage.id).label("message_count"))
            .outerjoin(ChatMessage)
            .where(ChatSession.user_id == user_id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.updated_at.desc())
        )
        rows = result.all()
        return [
            ChatSessionSchema(
                id=str(session.id),
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=count,
            )
            for session, count in rows
        ]

    async def get_session(
        self, session_id: str, user_id: uuid.UUID
    ) -> ChatSessionDetailSchema:
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        messages_result = await self.db.execute(
            sa.select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        msgs = messages_result.scalars().all()

        return ChatSessionDetailSchema(
            id=str(session.id),
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(msgs),
            messages=[
                ChatMessageSchema(
                    id=str(m.id),
                    role=m.role.value,
                    content=m.content,
                    sources=m.sources.get("items") if m.sources else None,
                    chart=m.chart,
                    created_at=m.created_at,
                )
                for m in msgs
            ],
        )

    async def update_session_title(
        self, session_id: str, user_id: uuid.UUID, title: str
    ) -> CreateChatSessionResponse:
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        session.title = title
        session.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        return CreateChatSessionResponse(session_id=str(session.id), title=session.title)

    async def delete_session(self, session_id: str, user_id: uuid.UUID) -> None:
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        await self.db.delete(session)
        await self.db.commit()

    async def send_message(
        self, session_id: str, user_id: uuid.UUID, content: str
    ) -> SendMessageResponse:
        session = await self.db.get(ChatSession, session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        user_msg = ChatMessage(
            session_id=session_id,
            role=ChatRole.USER,
            content=content,
        )
        self.db.add(user_msg)

        history = await self.db.execute(
            sa.select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        all_msgs = history.scalars().all()

        messages_payload = [{"role": m.role.value, "content": m.content} for m in all_msgs]

        try:
            llm_response = await self._chat_client.send_chat(
                messages=messages_payload,
                question=content,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")

        assistant_content = llm_response.get("summary", "")
        assistant_sources = llm_response.get("sources", [])
        assistant_chart = llm_response.get("chart")

        assistant_msg = ChatMessage(
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=assistant_content,
            sources={"items": assistant_sources} if assistant_sources else None,
            chart=assistant_chart,
        )
        self.db.add(assistant_msg)

        if session.title == "New Chat":
            session.title = (content[:80] + "...") if len(content) > 80 else content

        session.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(user_msg)
        await self.db.refresh(assistant_msg)

        return SendMessageResponse(
            user_message=ChatMessageSchema(
                id=str(user_msg.id),
                role=ChatRole.USER.value,
                content=user_msg.content,
                created_at=user_msg.created_at,
            ),
            assistant_message=ChatMessageSchema(
                id=str(assistant_msg.id),
                role=ChatRole.ASSISTANT.value,
                content=assistant_msg.content,
                sources=assistant_msg.sources.get("items")
                if assistant_msg.sources
                else None,
                chart=assistant_msg.chart,
                created_at=assistant_msg.created_at,
            ),
        )
