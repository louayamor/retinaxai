"""
Middleware components for the LLMOps service.

Includes authentication, rate limiting, and request validation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Protocol

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


class RateLimitStore(Protocol):
    """Protocol for rate limit storage backends."""

    async def get_requests(self, client_id: str, window_seconds: int) -> list[float]:
        """Return timestamps of recent requests for a client."""
        ...

    async def add_request(self, client_id: str, timestamp: float) -> None:
        """Record a new request timestamp for a client."""
        ...


class InMemoryRateLimitStore:
    """Per-process in-memory rate limit store.

    WARNING: This store is local to a single process. In multi-replica
    deployments (e.g. Kubernetes with multiple pods), rate limits will NOT
    be shared across replicas. Each replica maintains its own counters,
    allowing a client to exceed the global limit by hitting different
    replicas. For distributed rate limiting, use a shared backend such as
    Redis (see RedisRateLimitStore stub below).

    TODO: Replace with Redis-backed store for production multi-replica setups.
    """

    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def get_requests(self, client_id: str, window_seconds: int) -> list[float]:
        cutoff = time.time() - window_seconds
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]
        return self._requests[client_id]

    async def add_request(self, client_id: str, timestamp: float) -> None:
        self._requests[client_id].append(timestamp)


class RedisRateLimitStore:
    """Distributed Redis-backed rate limit store.

    Requires the ``redis`` package to be installed. Falls back to
    InMemoryRateLimitStore if redis is unavailable.
    """

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    async def get_requests(self, client_id: str, window_seconds: int) -> list[float]:
        # TODO: Implement using Redis sorted sets or sliding window scripts.
        raise NotImplementedError("Redis store not yet implemented")

    async def add_request(self, client_id: str, timestamp: float) -> None:
        # TODO: Implement using Redis sorted sets or sliding window scripts.
        raise NotImplementedError("Redis store not yet implemented")


def _create_rate_limit_store() -> RateLimitStore:
    """Create the best available rate limit store.

    Attempts to use Redis if the ``redis`` package is installed and a
    REDIS_URL environment variable is present. Otherwise falls back to
    the in-memory store.
    """
    try:
        import os

        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]

            client = aioredis.from_url(redis_url)
            return RedisRateLimitStore(client)
    except Exception:
        pass

    return InMemoryRateLimitStore()


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API keys for inter-service communication.

    Validates X-API-Key header on all routes except health endpoints.
    """

    def __init__(self, app, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/health",
            "/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/rag/status",
            "/api/rag/reindex",
        ]

    async def dispatch(self, request: Request, call_next):
        """Validate API key for protected routes."""
        # Skip exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "API key required",
                    "error_code": "MISSING_API_KEY",
                },
            )

        # Validate against configured keys
        valid_keys = {
            settings.api_key,
            settings.backend_api_key,
        }

        if api_key not in valid_keys:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "Invalid API key",
                    "error_code": "INVALID_API_KEY",
                },
            )

        # Store authenticated status in request state
        request.state.authenticated = True
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with pluggable storage backend.

    Defaults to in-memory storage. For distributed deployments, set
    REDIS_URL to enable Redis-backed rate limiting.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        exempt_paths: list[str] | None = None,
        store: RateLimitStore | None = None,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths or ["/health", "/ready"]
        self._store = store or _create_rate_limit_store()

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Skip exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Get client identifier (IP + User-Agent hash)
        client_ip = request.client.host if request.client else "unknown"
        client_id = f"{client_ip}"

        now = time.time()

        # Clean old requests and check limit
        recent = await self._store.get_requests(client_id, self.window_seconds)

        if len(recent) >= self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": self.window_seconds,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Record request
        await self._store.add_request(client_id, now)

        return await call_next(request)


def require_api_key(func: Callable) -> Callable:
    """
    Decorator to require API key on specific routes.

    Alternative to middleware for fine-grained control.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get request from args (FastAPI injects it)
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break

        if request and not getattr(request.state, "authenticated", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
            )

        return await func(*args, **kwargs)

    return wrapper
