from __future__ import annotations
import uuid
from datetime import UTC
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.auth.jwt_handler import create_access_token, create_refresh_token, decode_token
from app.auth.session_service import AuthSessionService
from app.core.config import settings
from app.core.exceptions import UnauthorizedException
from app.db.session import get_db
from app.models.auth_session import AuthSession
from app.schemas.token_schema import TokenResponse
from app.schemas.user_schema import UserCreate, UserRead
from app.users.service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_SAMESITE = "lax"
COOKIE_SECURE = settings.APP_ENV == "production"


def _token_response(access: str, refresh: str) -> JSONResponse:
    response = JSONResponse(
        content={
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "bearer",
        }
    )
    response.set_cookie(
        key="rxa_access_token",
        value=access,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="rxa_refresh_token",
        value=refresh,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
    )
    return response


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


@router.post("/register", response_model=UserRead, status_code=201)
async def register(
    data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = UserService(db)
    return await service.create(data)


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = UserService(db)
    user = await service.authenticate(form_data.username, form_data.password)
    access_token, jti = create_access_token(user.id)
    refresh_jti = str(uuid.uuid4())
    refresh_token = create_refresh_token(user.id, refresh_jti)
    session_service = AuthSessionService(db)
    await session_service.create_session(
        user.id, refresh_token, expires_in_days=7,
        access_token_jti=jti, refresh_token_jti=refresh_jti,
    )
    return _token_response(access_token, refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    data: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    refresh_token = request.cookies.get("rxa_refresh_token") or data.refresh_token
    if not refresh_token:
        raise UnauthorizedException("No refresh token provided.")

    payload = decode_token(refresh_token)
    if payload.token_type != "refresh":
        raise UnauthorizedException("Invalid refresh token.")

    session_service = AuthSessionService(db)
    if not await session_service.is_active(refresh_token):
        raise UnauthorizedException("Refresh token revoked or expired.")

    user_service = UserService(db)
    user = await user_service.get_by_id(uuid.UUID(payload.sub))
    access_token, new_jti = create_access_token(user.id)
    new_refresh_jti = str(uuid.uuid4())
    new_refresh_token = create_refresh_token(user.id, new_refresh_jti)
    await session_service.rotate_refresh_token(
        refresh_token, new_refresh_token, expires_in_days=7,
        new_access_jti=new_jti, new_refresh_jti=new_refresh_jti,
    )
    return _token_response(access_token, new_refresh_token)


@router.post("/logout")
async def logout(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    session_service = AuthSessionService(db)
    jti = getattr(request.state, "jti", None)

    if jti:
        session = await session_service.get_by_access_jti(jti)
        if session:
            await session_service.revoke_refresh_token(session.refresh_token)

    response = JSONResponse(content={"status": "ok"})
    response.delete_cookie(key="rxa_access_token", path="/")
    response.delete_cookie(key="rxa_refresh_token", path="/")
    return response


@router.post("/logout-all")
async def logout_all(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    await db.flush()
    response = JSONResponse(content={"status": "ok"})
    response.delete_cookie(key="rxa_access_token", path="/")
    response.delete_cookie(key="rxa_refresh_token", path="/")
    return response


@router.post("/cleanup-sessions")
async def cleanup_sessions(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from datetime import datetime, timezone

    await db.execute(
        delete(AuthSession).where(AuthSession.expires_at < datetime.now(UTC))
    )
    await db.flush()
    return {"status": "ok"}


@router.get("/me", response_model=UserRead)
async def get_current_user_me(user: CurrentUser):
    return UserRead.model_validate(user)
