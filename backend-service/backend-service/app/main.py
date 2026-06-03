from __future__ import annotations
from contextlib import asynccontextmanager
import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.api.v1.websockets import _listen_ws_broadcast, router as ws_router
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging
from app.core.middleware.cors import add_cors_middleware
from app.core.middleware.prometheus import PrometheusMiddleware
from app.core.middleware.rate_limit import RateLimitMiddleware
from app.core.middleware.request_id import RequestIDMiddleware
from app.core.prometheus_metrics import start_metrics_server
from app.observability.mlops_monitor import subscribe_mlops_monitor
from app.services.redis_client import redis_client as shared_redis
from app.services.task_tracker import bg_tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    start_metrics_server(port=9102)

    mlops_task = asyncio.create_task(subscribe_mlops_monitor())
    ws_broadcast_task = asyncio.create_task(_listen_ws_broadcast())

    if settings.APP_ENV == "development":
        _start_local_redis()

    yield

    await bg_tasks.drain(timeout=5.0)

    mlops_task.cancel()
    ws_broadcast_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await mlops_task
        await ws_broadcast_task

    await shared_redis.close()

    if settings.APP_ENV == "development":
        _stop_local_redis()


def _start_local_redis():
    import subprocess
    import time

    try:
        result = subprocess.run(
            ["redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            return
    except Exception:
        pass

    try:
        subprocess.Popen(
            ["redis-server", "--daemonize", "yes"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(10):
            time.sleep(0.5)
            result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and "PONG" in result.stdout:
                return
    except Exception:
        pass


def _stop_local_redis():
    pass


def _cors_headers(request: Request) -> dict[str, str]:
    origin = request.headers.get("origin")
    if origin in settings.CORS_ORIGINS:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        }
    return {}


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    add_cors_middleware(app)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        path_limits={
            "/api/v1/auth/login": (10, 60),
        },
    )
    app.add_middleware(PrometheusMiddleware)

    app.include_router(api_router)
    app.include_router(ws_router, tags=["websocket"])

    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/uploads",
        StaticFiles(directory=str(upload_dir)),
        name="uploads",
    )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_code": exc.error_code},
            headers=_cors_headers(request),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        import logging

        logging.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
            headers=_cors_headers(request),
        )

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    @app.get("/test-uploads/{path:path}")
    async def test_uploads(path: str):
        import os

        full_path = settings.data_dir / "uploads" / path
        exists = os.path.exists(full_path)
        return {"path": str(full_path), "exists": exists}

    return app


app = create_app()
