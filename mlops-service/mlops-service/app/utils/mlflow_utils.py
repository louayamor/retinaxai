from __future__ import annotations

import os
import time

import mlflow

import dagshub
from loguru import logger

from app.config.settings import Settings, settings

_RETRIES = 3
_RETRY_DELAY = 5.0


def configure_mlflow(s: Settings | None = None) -> None:
    s = s or settings

    for attempt in range(1, _RETRIES + 1):
        try:
            dagshub.init(
                repo_owner=s.dagshub_repo_owner,
                repo_name=s.dagshub_repo_name,
                mlflow=True,
            )
            mlflow.set_experiment("retinaxai-dr-classification")
            logger.info("mlflow configured via dagshub")
            return
        except Exception as e:
            if attempt < _RETRIES:
                logger.warning(
                    f"dagshub init attempt {attempt}/{_RETRIES} failed: {e}. "
                    f"retrying in {_RETRY_DELAY}s..."
                )
                time.sleep(_RETRY_DELAY)
            else:
                logger.warning(
                    f"dagshub unreachable after {_RETRIES} attempts: {e}. "
                    f"falling back to local mlruns/"
                )

    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
    mlflow.set_experiment("retinaxai-dr-classification-offline")
    logger.info("mlflow configured in offline mode (local mlruns/)")
