from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import AdminUser
from app.db.session import get_db
from app.models.auth_session import AuthSession
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.user_schema import UserRead, UserUpdate
from app.users.service import UserService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=dict)
async def list_users(
    _: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    role: str | None = Query(None),
):
    service = UserService(db)
    users = await service.list(skip=skip, limit=limit)
    if role:
        users = [u for u in users if u.role == role]
    total = await service.count()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [UserRead.model_validate(u) for u in users],
    }


@router.patch("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: uuid.UUID,
    data: UserUpdate,
    _: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = UserService(db)
    return await service.update(user_id, data)


@router.get("/stats", response_model=dict)
async def get_admin_stats(
    _: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    total_users = await db.scalar(select(func.count(User.id)))

    role_result = await db.execute(
        select(User.role, func.count(User.id)).group_by(User.role)
    )
    users_by_role = {row[0]: row[1] for row in role_result.all()}

    active_users = await db.scalar(
        select(func.count(User.id)).where(User.is_active.is_(True))
    )

    total_patients = await db.scalar(select(func.count(Patient.id)))
    total_predictions = await db.scalar(select(func.count(Prediction.id)))
    total_sessions = await db.scalar(select(func.count(AuthSession.id)))

    return {
        "users": {
            "total": total_users or 0,
            "by_role": users_by_role,
            "active": active_users or 0,
        },
        "platform": {
            "patients": total_patients or 0,
            "predictions": total_predictions or 0,
            "active_sessions": total_sessions or 0,
        },
    }
