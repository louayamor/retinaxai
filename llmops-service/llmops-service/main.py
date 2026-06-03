#!/usr/bin/env python
"""LLMOps CLI entry point."""

from __future__ import annotations

import argparse

import uvicorn
from loguru import logger

from app.pipeline.indexing_pipeline import IndexingPipeline


def serve() -> None:
    """Start the LLMOps API server."""
    from app.core.config import Settings

    settings = Settings()
    logger.info("Starting LLMOps API server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=False,
    )


def reindex() -> None:
    """Run the RAG reindexing pipeline."""
    logger.info("Starting RAG reindexing...")
    pipeline = IndexingPipeline()
    result = pipeline.run()
    logger.info(f"Reindexing complete: {result}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(description="RetinaXAI LLMOps Service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve")
    subparsers.add_parser("pipeline").add_argument(
        "--task", choices=["reindex"], default="reindex"
    )

    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "pipeline" and args.task == "reindex":
        reindex()

if __name__ == "__main__":
    main()
