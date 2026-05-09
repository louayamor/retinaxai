from __future__ import annotations

import mlflow

import dagshub
from loguru import logger

from app.config.settings import Settings, settings


def configure_mlflow(s: Settings | None = None) -> None:
    s = s or settings
    dagshub.init(
        repo_owner=s.dagshub_repo_owner,
        repo_name=s.dagshub_repo_name,
        mlflow=True,
    )
    mlflow.set_experiment("retinaxai-dr-classification")
    logger.info("mlflow configured via dagshub")
