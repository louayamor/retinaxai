from contextlib import asynccontextmanager

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

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
from app.services.monitoring.prometheus_metrics import (
    start_metrics_server,
    init_metrics,
    update_qwk_from_metrics_files,
    start_qwk_background_refresh,
    update_drift_metrics_from_files,
    start_drift_background_refresh,
    update_evaluation_metrics_from_files,
    start_evaluation_background_refresh,
)

logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


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
        from app.services.orchestration.automation_service import get_automation_service

        automation_service = get_automation_service(
            settings.artifacts_root,
            settings.artifacts_root / "monitoring" / "drift",
        )
        automation_service.start_scheduler(
            interval_hours=settings.automation_interval_hours
        )
    yield
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

    return app


app = create_app()
