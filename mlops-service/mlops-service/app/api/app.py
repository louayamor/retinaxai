import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import MLOpsException

from app.api.routes import (
    health,
    train,
    status,
    metrics,
    predict,
    rag,
    models,
    drift,
    features,
    automation,
    prometheus_proxy,
)
from app.api.routes import models_download
from app.api.dependencies import get_settings
from app.monitoring.prometheus_metrics import (
    start_metrics_server,
    init_metrics,
    update_qwk_from_metrics_files,
    start_qwk_background_refresh,
    update_drift_metrics_from_files,
    start_drift_background_refresh,
    update_evaluation_metrics_from_files,
    start_evaluation_background_refresh,
)
from app.monitoring.mlops_monitor_publisher import MLOpsMonitorPublisher

logger.disable("sqlalchemy.engine.Engine")
logger.disable("sqlalchemy.pool")

# Stage-module imports above replace loguru with JSON-serialized format;
# reset to human-readable for the API server.
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} - {message}",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"starting {settings.app_name} v{settings.app_version}")
    logger.info(f"environment: {settings.app_env}")
    init_metrics()
    start_metrics_server(port=settings.prometheus_metrics_port)

    # Bridge pipeline-evaluation metrics into Prometheus gauges
    update_qwk_from_metrics_files(settings.imaging_metrics_path)
    start_qwk_background_refresh(
        settings.imaging_metrics_path,
        interval_seconds=300,
    )
    update_evaluation_metrics_from_files(settings.imaging_metrics_path)
    start_evaluation_background_refresh(
        settings.imaging_metrics_path,
        interval_seconds=300,
    )

    # Bridge drift-check results into Prometheus gauges
    drift_history_path = settings.monitoring_dir / "drift" / "drift_history.json"
    evidently_metrics_path = settings.monitoring_dir / "evidently_metrics.json"
    update_drift_metrics_from_files(drift_history_path, evidently_metrics_path)
    start_drift_background_refresh(
        drift_history_path,
        evidently_metrics_path,
        interval_seconds=300,
    )

    if settings.automation_enabled:
        from app.monitoring.automation_service import get_automation_service

        automation_service = get_automation_service(
            settings.artifacts_root,
            settings.artifacts_root / "monitoring" / "drift",
        )
        automation_service.start_scheduler(
            interval_hours=settings.automation_interval_hours
        )
    monitor_publisher = MLOpsMonitorPublisher(
        redis_url=settings.redis_url,
        channel=settings.mlops_monitor_channel,
    )
    monitor_publisher.start(
        watch_paths=[
            settings.imaging_metrics_path,
            settings.imaging_artifacts_dir / "training_summary.json",
            settings.monitoring_dir / "drift" / "drift_history.json",
            settings.evidently_metrics_path,
        ]
    )
    yield
    await monitor_publisher.stop()
    logger.info("shutting down mlops service")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])  # type: ignore[arg-type]
    app.include_router(train.router, tags=["training"])  # type: ignore[arg-type]
    app.include_router(status.router, tags=["status"])  # type: ignore[arg-type]
    app.include_router(metrics.router, tags=["metrics"])  # type: ignore[arg-type]
    app.include_router(predict.router, tags=["predict"])  # type: ignore[arg-type]
    app.include_router(rag.router, tags=["rag"])  # type: ignore[arg-type]
    app.include_router(models.router, tags=["models"])  # type: ignore[arg-type]
    app.include_router(drift.router, tags=["drift"])  # type: ignore[arg-type]
    app.include_router(features.router, tags=["features"])  # type: ignore[arg-type]
    app.include_router(automation.router, tags=["automation"])  # type: ignore[arg-type]
    app.include_router(models_download.router, tags=["models"])  # type: ignore[arg-type]
    app.include_router(prometheus_proxy.router)  # type: ignore[arg-type]

    @app.exception_handler(MLOpsException)
    async def mlops_exception_handler(request: Request, exc: MLOpsException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_code": exc.error_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc.errors()),
                "error_code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.opt(exception=True).error("Unhandled exception: {}", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )

    return app


app = create_app()
