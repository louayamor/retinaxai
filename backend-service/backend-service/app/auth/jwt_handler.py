from __future__ import annotations
import uuid
from datetime import UTC, datetime, timedelta, timezone

import jwt

from app.core.config import get_settings, settings
from app.core.exceptions import UnauthorizedException
from app.schemas.token_schema import TokenPayload


def _create_token(
    subject: str, expire_delta: timedelta, token_type: str, jti: str | None = None
) -> str:
    expire = datetime.now(UTC) + expire_delta
    payload = {
        "sub": subject,
        "exp": int(expire.timestamp()),
        "token_type": token_type,
    }
    if jti:
        payload["jti"] = jti
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(user_id: uuid.UUID) -> tuple[str, str]:
    """Create access token with jti. Returns (token, jti)."""
    jti = str(uuid.uuid4())
    return (
        _create_token(
            str(user_id),
            timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
            "access",
            jti,
        ),
        jti,
    )


def create_refresh_token(user_id: uuid.UUID, jti: str | None = None) -> str:
    return _create_token(
        str(user_id),
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "refresh",
        jti,
    )


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return TokenPayload(**payload)
    except jwt.PyJWTError:
        raise UnauthorizedException()
