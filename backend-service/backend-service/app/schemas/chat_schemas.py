from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict[str, str]] | None = None
    chart: dict[str, Any] | None = None
    created_at: datetime


class ChatSessionSchema(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ChatSessionDetailSchema(ChatSessionSchema):
    messages: list[ChatMessageSchema] = []


class CreateChatSessionResponse(BaseModel):
    session_id: str
    title: str


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=5000)


class SendMessageResponse(BaseModel):
    user_message: ChatMessageSchema
    assistant_message: ChatMessageSchema


class UpdateSessionTitle(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSchema]
    total: int
