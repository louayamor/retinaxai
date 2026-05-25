from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from app.core.config import settings


class RedisClient:
    _instance: RedisClient | None = None
    _redis: aioredis.Redis | None = None
    _initialized = False

    def __new__(cls) -> RedisClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_connection(self) -> aioredis.Redis | None:
        if self._redis is not None:
            try:
                await self._redis.ping()
                return self._redis
            except Exception:
                self._redis = None

        try:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await self._redis.ping()
            logger.info("Shared Redis connection established")
            return self._redis
        except Exception as e:
            logger.warning("Redis unavailable", error=str(e))
            self._redis = None
            return None

    @staticmethod
    async def create_dedicated_connection() -> aioredis.Redis | None:
        try:
            redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=None,
            )
            await redis.ping()
            return redis
        except Exception as e:
            logger.warning("Redis dedicated connection failed", error=str(e))
            return None

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
            logger.info("Shared Redis connection closed")


redis_client = RedisClient()
