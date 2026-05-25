from __future__ import annotations
import uuid
from typing import Annotated, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.auth.dependencies import CurrentUser
from app.db.session import get_db
from app.models.notification import Notification


router = APIRouter(prefix="/notifications", tags=["notifications"])


class NotificationResponse(BaseModel):
    id: uuid.UUID
    type: str
    title: str
    message: str
    read: bool
    created_at: datetime
    user_id: Optional[uuid.UUID] = None

    class Config:
        from_attributes = True


class MarkReadRequest(BaseModel):
    notification_ids: list[uuid.UUID]


@router.get("/", response_model=list[NotificationResponse])
async def list_notifications(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    unread_only: bool = Query(False, description="Filter only unread notifications"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get all notifications for the current user."""
    user_id = current_user.id

    query = (
        select(Notification)
        .where((Notification.user_id == user_id) | (Notification.user_id.is_(None)))
        .order_by(Notification.created_at.desc())
    )

    if unread_only:
        query = query.where(Notification.read.is_(False))

    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    notifications = result.scalars().all()

    return notifications


@router.get("/unread-count")
async def get_unread_count(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get count of unread notifications."""
    user_id = current_user.id

    from sqlalchemy import func

    query = select(func.count(Notification.id)).where(
        ((Notification.user_id == user_id) | (Notification.user_id.is_(None)))
        & (not Notification.read)
    )

    result = await db.execute(query)
    count = result.scalar() or 0

    return {"unread_count": count}


@router.put("/read")
async def mark_notifications_read(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    request: MarkReadRequest,
):
    """Mark specific notifications as read."""
    user_id = current_user.id

    stmt = (
        update(Notification)
        .where(
            Notification.id.in_(request.notification_ids),
            (Notification.user_id == user_id) | (Notification.user_id.is_(None)),
        )
        .values(read=True)
    )

    await db.execute(stmt)
    await db.commit()

    return {"status": "ok", "marked_count": len(request.notification_ids)}


@router.put("/read-all")
async def mark_all_read(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Mark all notifications as read."""
    user_id = current_user.id

    stmt = (
        update(Notification)
        .where((Notification.user_id == user_id) | (Notification.user_id.is_(None)))
        .values(read=True)
    )

    result = await db.execute(stmt)
    await db.commit()

    return {"status": "ok", "marked_count": result.rowcount}


@router.delete("/clear")
async def clear_notifications(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    read_only: bool = Query(True, description="Clear only read notifications"),
):
    """Clear notifications (all or read only)."""
    user_id = current_user.id

    if read_only:
        stmt = delete(Notification).where(
            ((Notification.user_id == user_id) | (Notification.user_id.is_(None)))
            & (Notification.read)
        )
    else:
        stmt = delete(Notification).where(
            (Notification.user_id == user_id) | (Notification.user_id.is_(None))
        )

    result = await db.execute(stmt)
    await db.commit()

    return {"status": "ok", "deleted_count": result.rowcount}
