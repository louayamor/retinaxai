from __future__ import annotations

import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import AdminUser
from app.core.config import settings
from app.db.session import get_db
from app.models.auth_session import AuthSession
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.user_schema import UserCreate, UserRead, UserUpdate
from app.services.redis_client import redis_client
from app.users.service import UserService

router = APIRouter(prefix="/admin", tags=["admin"])


async def _ping_service(url: str, timeout: float = 3.0) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/health")
            return "healthy" if resp.status_code < 500 else "unhealthy"
    except Exception:
        return None


@router.get("/health")
async def get_admin_health(_: AdminUser):
    redis_ok = None
    try:
        conn = await redis_client.get_connection()
        if conn:
            await conn.ping()
            redis_ok = "healthy"
    except Exception:
        redis_ok = None

    pg_ok = None
    try:
        from app.db.session import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            pg_ok = "healthy"
    except Exception:
        pg_ok = None

    mlops_status = await _ping_service(settings.ml_service_url)
    llmops_status = await _ping_service(settings.llm_service_url)

    return {
        "backend": "healthy",
        "mlops": mlops_status,
        "llmops": llmops_status,
        "redis": redis_ok,
        "postgres": pg_ok,
    }


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


@router.post("/users", response_model=UserRead, status_code=201)
async def create_user(
    data: UserCreate,
    _: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = UserService(db)
    return await service.create(data)


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
