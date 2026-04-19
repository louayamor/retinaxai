"""
RetinaXAI LLMOps Service - Main Application Entry Point.

This module initializes the FastAPI application with proper lifespan management,
middleware, exception handling, and service dependencies.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

try:
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
except Exception:
    pass

from app.api.routes import router
from app.core.config import settings
from app.core.middleware import APIKeyMiddleware, RateLimitMiddleware
from app.pipeline.report_generator import generate_report_handler
from app.services.job_manager import get_job_manager
from app.services.operation_state import set_operation
from app.services.prometheus_metrics import start_metrics_server
from app.vectorstore.chroma_store import ChromaStore

# Track startup time for health checks
_STARTUP_TIME: float | None = None


def configure_mlflow() -> bool:
    """
    Configure MLflow with Dagshub for experiment tracking.

    Returns:
        bool: True if MLflow was successfully configured, False otherwise.
    """
    if not settings.mlflow_tracking_uri:
        logger.info("MLflow tracking URI not configured, skipping MLflow setup")
        return False

    try:
        import dagshub
        import mlflow

        dagshub.init(
            repo_owner="louayamor",
            repo_name="retinaxai",
            mlflow=True,
        )
        mlflow.set_experiment(settings.mlflow_experiment_name)
        logger.info(f"MLflow configured via DagsHub: {settings.mlflow_experiment_name}")
        return True
    except ModuleNotFoundError:
        logger.warning("dagshub or mlflow not installed, skipping MLflow setup")
        return False
    except Exception as e:
        logger.error(f"Failed to configure MLflow: {e}")
        return False


def validate_directories() -> dict[str, bool]:
    """
    Validate and create required directories.

    Returns:
        dict[str, bool]: Mapping of directory names to their creation status.
    """
    dirs_to_validate = {
        "data_dir": settings.data_dir,
        "cache_dir": settings.cache_dir,
        "chroma_persist": settings.rag_chroma_persist_directory,
    }

    results = {}
    for name, dir_path in dirs_to_validate.items():
        try:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            # Verify directory is writable
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            results[name] = True
            logger.info(f"Directory validated: {path}")
        except Exception as e:
            results[name] = False
            logger.error(f"Directory validation failed for {name} ({dir_path}): {e}")

    return results


def check_chromadb_ready() -> bool:
    """
    Check if ChromaDB is accessible and ready.

    Returns:
        bool: True if ChromaDB is ready, False otherwise.
    """
    try:
        store = ChromaStore(
            settings.rag_chroma_persist_directory,
            settings.rag_chroma_collection_name,
            settings.rag_embedding_model,
        )
        store.ensure_ready()
        logger.info("ChromaDB is ready")
        return True
    except Exception as e:
        logger.warning(f"ChromaDB not ready: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown logic, including:
    - Directory validation
    - MLflow configuration
    - ChromaDB readiness check
    - Service initialization logging
    """
    global _STARTUP_TIME
    _STARTUP_TIME = time.time()

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")

    # Start Prometheus metrics server
    start_metrics_server(port=settings.prometheus_metrics_port)

    # Validate directories
    dir_results = validate_directories()
    if not all(dir_results.values()):
        failed = [name for name, ok in dir_results.items() if not ok]
        logger.error(f"Directory validation failed for: {', '.join(failed)}")
        raise RuntimeError(f"Failed to validate directories: {failed}")

    # Configure MLflow
    mlflow_ready = configure_mlflow()
    logger.info(f"MLflow status: {'ready' if mlflow_ready else 'not configured'}")

    # Check ChromaDB
    chroma_ready = check_chromadb_ready()
    logger.info(f"ChromaDB status: {'ready' if chroma_ready else 'not ready'}")

    # Initialize job manager
    job_manager = get_job_manager()
    job_manager.register_handler("report_generation", generate_report_handler)
    await job_manager.start()
    logger.info(
        f"Job manager started with handlers: {list(job_manager._handlers.keys())}"
    )

    logger.info(f"Service ready on {settings.app_host}:{settings.app_port}")

    yield

    # Stop job manager on shutdown
    await job_manager.stop()

    # Shutdown logic
    shutdown_duration = time.time() - _STARTUP_TIME if _STARTUP_TIME else 0
    logger.info(f"Shutting down {settings.app_name} (uptime: {shutdown_duration:.2f}s)")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured application instance.
    """
    import logging
    logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="LLMOps Service for RetinaXAI - RAG-powered medical report generation",
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Key authentication middleware
    app.add_middleware(
        APIKeyMiddleware,
        exempt_paths=[
            "/health",
            "/ready",
            "/api/health",
            "/api/ready",
            "/api/rag",
            "/api/generate",
            "/api/operation",
            "/api/xai/explain",
            "/api/xai/gradcam",
            "/api/xai/severity",
            "/api/xai/shap/explain",
            "/api/xai/shap/importance",
            "/api/xai/shap/bias",
            "/api/workflows/training-complete",
            "/api/jobs",
            "/api/jobs/",
            "/api/operation/status",
        ],
    )

    # Rate limiting middleware
    if settings.enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )

    # Request ID middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Add request ID for tracing across services."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        response.headers["X-Request-ID"] = request_id
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} "
            f"({duration:.3f}s) [{request_id[:8]}]"
        )
        return response

    # Include routers
    app.include_router(router)  # type: ignore[arg-type]

    # Exception handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"Unhandled exception [{request_id}]: {exc}", exc_info=True)
        set_operation("error", str(exc)[:200])
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc)[:200],
                "error_code": "INTERNAL_ERROR",
                "request_id": request_id,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"Validation error [{request_id}]: {exc}")
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "error_code": "VALIDATION_ERROR",
                "request_id": request_id,
            },
        )

    # Health check endpoints
    @app.get("/health", tags=["health"])
    async def health():
        """
        Basic health check.

        Returns service status and version info.
        """
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "llm_provider": settings.llm_provider.value,
            "model": settings.llm_model,
        }

    @app.get("/ready", tags=["health"])
    async def ready():
        """
        Readiness check for Kubernetes.

        Returns 200 only when all dependencies are ready.
        """
        checks = {
            "directories": True,  # Already validated at startup
            "chromadb": check_chromadb_ready(),
        }

        if not all(checks.values()):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "checks": checks,
                },
            )

        return {
            "status": "ready",
            "checks": checks,
        }

    return app


def run_serve() -> None:
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    import uvicorn

    logger.info("Starting LLMOps API server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)


def run_reindex() -> None:
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    from app.pipeline.indexing_pipeline import IndexingPipeline

    logger.info("Starting RAG reindexing...")
    pipeline = IndexingPipeline()
    result = pipeline.run()
    logger.info(f"Reindexing complete: {result}")


def main():
    import argparse
    import os
    import sys
    from pathlib import Path

    base = Path(__file__).parent
    os.chdir(base)
    sys.path.insert(0, str(base))

    parser = argparse.ArgumentParser(description="RetinaXAI LLMOps Service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve")
    subparsers.add_parser("pipeline").add_argument(
        "--task", choices=["reindex"], default="reindex"
    )
    args = parser.parse_args()

    if args.command == "serve":
        run_serve()
    elif args.command == "pipeline" and args.task == "reindex":
        run_reindex()


# Create application instance
app = create_app()

if __name__ == "__main__":
    main()
