from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import decode_token
from app.services.redis_client import redis_client as shared_redis
from app.core.exceptions import UnauthorizedException
from app.db.session import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user_from_cookie(request: Request) -> str:
    """Extract access token from cookie."""
    token = request.cookies.get("rxa_access_token")
    if not token:
        raise UnauthorizedException("Not authenticated")
    return token


async def get_current_user(
    token: Annotated[str, Depends(get_current_user_from_cookie)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    payload = decode_token(token)
    if payload.token_type != "access":
        raise UnauthorizedException()

    # Validate session still exists in Redis
    jti = getattr(payload, "jti", None)
    if jti:
        redis = await shared_redis.get_connection()
        if redis:
            # Check if access token jti is still valid (session not revoked)
            from app.auth.session_service import AuthSessionService

            session = await AuthSessionService(db).get_by_access_jti(jti)
            if not session:
                raise UnauthorizedException("Session expired or revoked")

    user = await db.get(User, uuid.UUID(payload.sub))
    if not user or not user.is_active:
        raise UnauthorizedException()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
