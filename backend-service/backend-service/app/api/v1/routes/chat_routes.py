from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.db.session import get_db
from app.models.chat import ChatMessage, ChatRole, ChatSession
from app.schemas.chat_schemas import (
    ChatMessageSchema,
    ChatSessionDetailSchema,
    ChatSessionListResponse,
    ChatSessionSchema,
    CreateChatSessionResponse,
    SendMessageRequest,
    SendMessageResponse,
    UpdateSessionTitle,
)
from app.services.llm_client.chat_client import chat_client

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=CreateChatSessionResponse, status_code=201)
async def create_session(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateChatSessionResponse:
    session = ChatSession(user_id=current_user.id, title="New Chat")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return CreateChatSessionResponse(session_id=str(session.id), title=session.title)


@router.get("/sessions", response_model=ChatSessionListResponse)
async def list_sessions(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ChatSessionListResponse:
    result = await db.execute(
        sa.select(
            ChatSession,
            sa.func.count(ChatMessage.id).label("message_count"),
        )
        .outerjoin(ChatMessage)
        .where(ChatSession.user_id == current_user.id)
        .group_by(ChatSession.id)
        .order_by(ChatSession.updated_at.desc())
    )
    rows = result.all()

    sessions = [
        ChatSessionSchema(
            id=str(session.id),
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=count,
        )
        for session, count in rows
    ]
    return ChatSessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailSchema)
async def get_session(
    session_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ChatSessionDetailSchema:
    session = await db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    messages_result = await db.execute(
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


@router.patch("/sessions/{session_id}", response_model=CreateChatSessionResponse)
async def update_session_title(
    session_id: str,
    body: UpdateSessionTitle,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CreateChatSessionResponse:
    session = await db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    session.title = body.title
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return CreateChatSessionResponse(session_id=str(session.id), title=session.title)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    session = await db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
    return {"status": "deleted"}


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SendMessageResponse,
    status_code=201,
)
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> SendMessageResponse:
    session = await db.get(ChatSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = ChatMessage(
        session_id=session_id,
        role=ChatRole.USER,
        content=body.content,
    )
    db.add(user_msg)

    history = await db.execute(
        sa.select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    all_msgs = history.scalars().all()

    messages_payload = [{"role": m.role.value, "content": m.content} for m in all_msgs]

    try:
        llm_response = await chat_client.send_chat(
            messages=messages_payload,
            question=body.content,
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
    db.add(assistant_msg)

    if session.title == "New Chat":
        session.title = (
            (body.content[:80] + "...") if len(body.content) > 80 else body.content
        )

    session.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user_msg)
    await db.refresh(assistant_msg)

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
