from __future__ import annotations
import json
import uuid
from datetime import UTC, datetime, timedelta, timezone

import redis.asyncio as aioredis
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedException
from app.models.auth_session import AuthSession

class _RedisManager:
    def __init__(self) -> None:
        self.connection: aioredis.Redis | None = None


_redis_manager = _RedisManager()


async def get_session_redis() -> aioredis.Redis | None:
    if _redis_manager.connection is None:
        settings = get_settings()

        redis_url = settings.REDIS_URL

        # Check for optional env var to disable Redis entirely
        if settings.APP_ENV == "development":
            import os

            if os.environ.get("DISABLE_REDIS"):
                logger.info("Redis disabled via DISABLE_REDIS env var")
                return None

        redis_url = settings.REDIS_URL

        try:
            _redis_manager.connection = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            ping_result = await _redis_manager.connection.ping()
            if ping_result:
                logger.info("Session Redis connection established")
        except Exception as e:
            import os

            app_env = getattr(
                settings, "APP_ENV", os.environ.get("APP_ENV", "production")
            )
            if app_env == "production":
                logger.warning(f"Redis not available for sessions: {e}")
            else:
                logger.info(
                    f"Redis not available (running locally without Docker?): {e}"
                )
            _redis_manager.connection = None
    return _redis_manager.connection


class AuthSessionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(
        self,
        user_id: uuid.UUID,
        refresh_token: str,
        expires_in_days: int,
        access_token_jti: str | None = None,
    ) -> AuthSession:
        session = AuthSession(
            user_id=user_id,
            access_token_jti=access_token_jti,
            refresh_token=refresh_token,
            expires_at=datetime.now(UTC) + timedelta(days=expires_in_days),
        )
        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)

        redis = await get_session_redis()
        if redis:
            try:
                key = f"session:{refresh_token}"
                ttl = expires_in_days * 86400
                session_data = {
                    "user_id": str(session.user_id),
                    "session_id": str(session.id),
                    "access_token_jti": access_token_jti,
                    "created_at": session.created_at.isoformat(),
                    "expires_at": session.expires_at.isoformat(),
                }
                await redis.setex(key, ttl, json.dumps(session_data))
                logger.debug(f"Session cached in Redis: {key}")
            except Exception as e:
                logger.warning(f"Failed to cache session in Redis: {e}")

        return session

    async def get_by_refresh_token(self, refresh_token: str) -> AuthSession | None:
        redis = await get_session_redis()
        if redis:
            try:
                key = f"session:{refresh_token}"
                cached = await redis.get(key)
                if cached:
                    data = json.loads(cached)
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    if expires_at > datetime.now(UTC):
                        logger.debug(f"Session found in Redis cache: {key}")
                        result = await self.db.execute(
                            select(AuthSession).where(
                                AuthSession.refresh_token == refresh_token
                            )
                        )
                        return result.scalar_one_or_none()
                    else:
                        await redis.delete(key)
                        logger.debug(f"Session expired in Redis, removed: {key}")
            except Exception as e:
                logger.warning(f"Redis cache lookup failed: {e}")

        result = await self.db.execute(
            select(AuthSession).where(AuthSession.refresh_token == refresh_token)
        )
        return result.scalar_one_or_none()

    async def is_active(self, refresh_token: str) -> bool:
        redis = await get_session_redis()
        if redis:
            try:
                key = f"session:{refresh_token}"
                cached = await redis.get(key)
                if cached:
                    data = json.loads(cached)
                    expires_at = datetime.fromisoformat(data["expires_at"])
                    return expires_at > datetime.now(UTC)
            except Exception as e:
                logger.warning(f"Redis active check failed: {e}")

        session = await self.get_by_refresh_token(refresh_token)
        if not session or session.revoked:
            return False
        return session.expires_at > datetime.now(UTC)

    async def revoke_refresh_token(self, refresh_token: str) -> None:
        session = await self.get_by_refresh_token(refresh_token)
        if session:
            session.revoked = True
            await self.db.flush()

        redis = await get_session_redis()
        if redis:
            try:
                key = f"session:{refresh_token}"
                await redis.delete(key)
                logger.debug(f"Session revoked in Redis: {key}")
            except Exception as e:
                logger.warning(f"Failed to revoke session in Redis: {e}")

    async def rotate_refresh_token(
        self,
        old_refresh_token: str,
        new_refresh_token: str,
        expires_in_days: int,
        new_access_jti: str | None = None,
    ) -> str:
        session = await self.get_by_refresh_token(old_refresh_token)
        if not session or session.revoked:
            raise UnauthorizedException("Refresh token revoked or expired.")
        session.revoked = True
        await self.create_session(
            session.user_id, new_refresh_token, expires_in_days, new_access_jti
        )

        redis = await get_session_redis()
        if redis:
            try:
                old_key = f"session:{old_refresh_token}"
                await redis.delete(old_key)
                logger.debug(f"Old session removed from Redis: {old_key}")
            except Exception as e:
                logger.warning(f"Failed to clean old session in Redis: {e}")

        return new_refresh_token

    async def get_by_access_jti(self, jti: str) -> AuthSession | None:
        """Find session by access token jti (for validation on protected routes)."""
        redis = await get_session_redis()
        if redis:
            try:
                keys = []
                async for key in redis.scan_iter(match="session:*"):
                    keys.append(key)
                if keys:
                    cached = await redis.mget(keys)
                    for key, data in zip(keys, cached):
                        if data:
                            session_data = json.loads(data)
                            if session_data.get("access_token_jti") == jti:
                                return await self._load_session_from_db(key)
            except Exception as e:
                logger.warning(f"Redis session lookup failed: {e}")

        stmt = select(AuthSession).where(
            AuthSession.access_token_jti == jti,
            AuthSession.revoked.is_(False),
            AuthSession.expires_at > datetime.now(UTC),
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _load_session_from_db(self, redis_key: str) -> AuthSession | None:
        """Load session from DB using refresh token extracted from Redis key."""
        refresh_token = redis_key.replace("session:", "")
        return await self.get_by_refresh_token(refresh_token)
