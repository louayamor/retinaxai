from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.repository import ChatRepository
from app.models.chat import ChatMessage, ChatRole
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
        self.repo = ChatRepository(db)
        self.db = db
        self._chat_client = chat_client_override or chat_client

    async def create_session(self, user_id: uuid.UUID) -> CreateChatSessionResponse:
        session = await self.repo.create_session(user_id)
        return CreateChatSessionResponse(session_id=str(session.id), title=session.title)

    async def list_sessions(self, user_id: uuid.UUID) -> list[ChatSessionSchema]:
        rows = await self.repo.list_user_sessions(user_id)
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
        session = await self.repo.get_session(session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        msgs = await self.repo.get_session_messages(session_id)

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
        session = await self.repo.get_session(session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        session.title = title
        session.updated_at = datetime.now(timezone.utc)
        await self.repo.update_session(session)
        return CreateChatSessionResponse(session_id=str(session.id), title=session.title)

    async def delete_session(self, session_id: str, user_id: uuid.UUID) -> None:
        session = await self.repo.get_session(session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        await self.repo.delete_session(session)

    async def send_message(
        self, session_id: str, user_id: uuid.UUID, content: str
    ) -> SendMessageResponse:
        session = await self.repo.get_session(session_id)
        if not session or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")

        user_msg = ChatMessage(
            session_id=session_id,
            role=ChatRole.USER,
            content=content,
        )
        await self.repo.add_message(user_msg)

        all_msgs = await self.repo.get_session_messages(session_id)
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
        await self.repo.add_message(assistant_msg)

        if session.title == "New Chat":
            session.title = (content[:80] + "...") if len(content) > 80 else content

        session.updated_at = datetime.now(timezone.utc)
        await self.repo.update_session(session)

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
                sources=assistant_sources,
                chart=assistant_chart,
                created_at=assistant_msg.created_at,
            ),
        )
