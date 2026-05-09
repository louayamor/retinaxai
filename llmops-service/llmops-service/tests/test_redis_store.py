from __future__ import annotations


import pytest
from app.core.middleware import RedisRateLimitStore


@pytest.mark.asyncio
async def test_redis_store_get_requests_raises_not_implemented():
    store = RedisRateLimitStore(None)
    with pytest.raises(NotImplementedError):
        await store.get_requests("client-1", 60)


@pytest.mark.asyncio
async def test_redis_store_add_request_raises_not_implemented():
    store = RedisRateLimitStore(None)
    with pytest.raises(NotImplementedError):
        await store.add_request("client-1", 1000.0)
