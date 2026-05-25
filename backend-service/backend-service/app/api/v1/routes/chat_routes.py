from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import DoctorUser
from app.chat.service import ChatService
from app.db.session import get_db
from app.schemas.chat_schemas import (
    ChatSessionDetailSchema,
    ChatSessionListResponse,
    CreateChatSessionResponse,
    SendMessageRequest,
    SendMessageResponse,
    UpdateSessionTitle,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=CreateChatSessionResponse, status_code=201)
async def create_session(
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> CreateChatSessionResponse:
    service = ChatService(db)
    return await service.create_session(current_user.id)


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_sessions(
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> ChatSessionListResponse:
    service = ChatService(db)
    sessions = await service.list_sessions(current_user.id)
    return ChatSessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailSchema)
async def get_session(
    session_id: str,
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> ChatSessionDetailSchema:
    service = ChatService(db)
    return await service.get_session(session_id, current_user.id)


@router.patch("/sessions/{session_id}", response_model=CreateChatSessionResponse)
async def update_session_title(
    session_id: str,
    body: UpdateSessionTitle,
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> CreateChatSessionResponse:
    service = ChatService(db)
    return await service.update_session_title(session_id, current_user.id, body.title)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    service = ChatService(db)
    await service.delete_session(session_id, current_user.id)
    return {"status": "deleted"}


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=201,
)
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    current_user: DoctorUser,
    db: AsyncSession = Depends(get_db),
) -> SendMessageResponse:
    service = ChatService(db)
    return await service.send_message(session_id, current_user.id, body.content)
