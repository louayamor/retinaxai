from __future__ import annotations

import asyncio
import time

from fastapi import Request
from starlette.datastructures import URL, Headers

from app.core.middleware import RateLimitMiddleware


class DummyApp:
    pass


def _make_request(
    path: str = "/api/generate", client_host: str = "127.0.0.1"
) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_rate_limit_middleware_allows_requests_under_limit():
    async def call_next(request: Request):
        return {"status": "ok"}

    middleware = RateLimitMiddleware(DummyApp(), max_requests=3, window_seconds=60)

    req = _make_request()
    for _ in range(3):
        response = asyncio.run(middleware.dispatch(req, call_next))
        assert response == {"status": "ok"}


def test_rate_limit_middleware_blocks_over_limit():
    async def call_next(request: Request):
        return {"status": "ok"}

    middleware = RateLimitMiddleware(DummyApp(), max_requests=2, window_seconds=60)

    req = _make_request()
    asyncio.run(middleware.dispatch(req, call_next))
    asyncio.run(middleware.dispatch(req, call_next))

    response = asyncio.run(middleware.dispatch(req, call_next))
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_rate_limit_middleware_skips_exempt_paths():
    async def call_next(request: Request):
        return {"status": "ok"}

    middleware = RateLimitMiddleware(
        DummyApp(), max_requests=0, window_seconds=60, exempt_paths=["/health"]
    )

    req = _make_request("/health")
    response = asyncio.run(middleware.dispatch(req, call_next))
    assert response == {"status": "ok"}


def test_rate_limit_middleware_cleans_old_requests():
    async def call_next(request: Request):
        return {"status": "ok"}

    middleware = RateLimitMiddleware(DummyApp(), max_requests=2, window_seconds=1)

    req = _make_request()
    asyncio.run(middleware.dispatch(req, call_next))
    asyncio.run(middleware.dispatch(req, call_next))

    time.sleep(1.1)

    response = asyncio.run(middleware.dispatch(req, call_next))
    assert response == {"status": "ok"}


def test_rate_limit_middleware_tracks_multiple_clients():
    async def call_next(request: Request):
        return {"status": "ok"}

    middleware = RateLimitMiddleware(DummyApp(), max_requests=1, window_seconds=60)

    req_a = _make_request(client_host="1.2.3.4")
    req_b = _make_request(client_host="5.6.7.8")

    asyncio.run(middleware.dispatch(req_a, call_next))
    asyncio.run(middleware.dispatch(req_b, call_next))

    response_a = asyncio.run(middleware.dispatch(req_a, call_next))
    assert response_a.status_code == 429

    response_b = asyncio.run(middleware.dispatch(req_b, call_next))
    assert response_b.status_code == 429
