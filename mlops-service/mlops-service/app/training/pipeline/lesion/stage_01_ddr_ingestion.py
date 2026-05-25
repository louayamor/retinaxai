from __future__ import annotations

import sys
from pathlib import Path

import dagshub
import mlflow
from loguru import logger

from app.constants import PARAMS_FILE_PATH
from app.training.components.ddr_ingestion import DDRDataIngestion
from app.utils.common import read_yaml

logger.remove()
logger.add(sys.stdout, serialize=True)


def run() -> None:
    logger.info(">>> stage 01: DDR ingestion started")

    params = read_yaml(PARAMS_FILE_PATH)
    lesion_cfg = params.get("lesion_model", {})
    target_dir = Path(str(lesion_cfg.get("ddr_target_dir", "data/lesions/ddr")))

    dagshub.init(repo_owner="louayamor", repo_name="retinaxai", mlflow=True)

    with mlflow.start_run(run_name="stage_01_ddr_ingestion"):
        ingestion = DDRDataIngestion(target_dir)
        result_path = ingestion.run()

        mlflow.log_param("ddr_repo", str(lesion_cfg.get("ddr_hf_repo", "")))
        mlflow.log_param("ddr_target_dir", str(result_path))
        mlflow.log_artifact(str(result_path), artifact_path="ddr_data")

        logger.info(f"stage 01 complete: {result_path}")


if __name__ == "__main__":
    run()
