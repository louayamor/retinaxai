"""
Main entry point for the RetinaXAI Biomarker Service.

This module initializes the FastAPI application and handles startup/shutdown events.
"""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from biomarker_service.api.routes import create_api_router
from biomarker_service.infrastructure.model_registry import initialize_model

# Set environment variables for model caching
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["RTNLS_MODEL_RELEASES"] = "/app/.cache/retinalysis-models"

# Ensure cache directories exist
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["RTNLS_MODEL_RELEASES"]).mkdir(parents=True, exist_ok=True)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RetinaXAI Biomarker Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Include API routes
    app.include_router(create_api_router())

    @app.on_event("startup")
    async def startup_event():
        """Initialize the VascX model on startup."""
        logger.info("Initializing VascX model...")
        await initialize_model()
        logger.info("VascX model initialized successfully.")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        logger.info("Shutting down biomarker service...")

    return app


def main() -> None:
    """Entry point for the CLI command 'biomarker-service'."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        "biomarker_service.main:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
    )


if __name__ == "__main__":
    main()
