"""
Middleware components for the LLMOps service.

Includes authentication, rate limiting, and request validation.
"""
from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps
from typing import Callable

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


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
    Simple in-memory rate limiting middleware.

    Limits requests per client IP address.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        exempt_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths or ["/health", "/ready"]
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Skip exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Get client identifier (IP + User-Agent hash)
        client_ip = request.client.host if request.client else "unknown"
        client_id = f"{client_ip}"

        now = time.time()

        # Clean old requests
        cutoff = now - self.window_seconds
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
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
        self._requests[client_id].append(now)

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
