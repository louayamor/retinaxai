from __future__ import annotations
from unittest.mock import AsyncMock

import pytest

from app.api.v1.routes.auth_routes import RefreshRequest, refresh_token
from app.auth.jwt_handler import decode_token
from app.auth.session_service import AuthSessionService
from app.core.exceptions import UnauthorizedException


def test_create_access_token_marks_type_access(access_token: str) -> None:
    payload = decode_token(access_token)

    assert payload.token_type == "access"


def test_create_refresh_token_marks_type_refresh(refresh_token: str) -> None:
    payload = decode_token(refresh_token)

    assert payload.token_type == "refresh"


@pytest.mark.asyncio
async def test_refresh_rejects_access_token(access_token: str, auth_db_session) -> None:
    with pytest.raises(UnauthorizedException):
        await refresh_token(RefreshRequest(refresh_token=access_token), auth_db_session)


@pytest.mark.asyncio
async def test_logout_marks_session_revoked(
    auth_db_session, auth_session, refresh_token: str
) -> None:
    session_service = AuthSessionService(auth_db_session)

    auth_db_session.execute = AsyncMock(return_value=MockResult(auth_session))
    auth_db_session.flush = AsyncMock()

    await session_service.revoke_refresh_token(refresh_token)
    assert auth_session.revoked is True


class MockResult:
    def __init__(self, obj):
        self.obj = obj

    def scalar_one_or_none(self):
        return self.obj
