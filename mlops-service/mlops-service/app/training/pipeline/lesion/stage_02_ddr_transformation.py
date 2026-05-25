from __future__ import annotations

import sys
from pathlib import Path

import dagshub
import mlflow
from loguru import logger

from app.constants import PARAMS_FILE_PATH
from app.training.components.ddr_transformation import DDRTransformation
from app.utils.common import read_yaml

logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = Path("artifacts")


def run() -> None:
    logger.info(">>> stage 02: DDR transformation started")

    params = read_yaml(PARAMS_FILE_PATH)
    lesion_cfg = params.get("lesion_model", {})
    ddr_target = Path(str(lesion_cfg.get("ddr_target_dir", "data/lesions/ddr")))
    ddr_seg_dir = ddr_target / "lesion_segmentation"
    output_dir = ARTIFACTS_DIR / "lesion" / "manifests"

    dagshub.init(repo_owner="louayamor", repo_name="retinaxai", mlflow=True)

    with mlflow.start_run(run_name="stage_02_ddr_transformation"):
        transformer = DDRTransformation(ddr_seg_dir, output_dir)
        manifests = transformer.run()

        for split, csv_path in manifests.items():
            mlflow.log_param(f"{split}_manifest", str(csv_path))
            mlflow.log_artifact(str(csv_path), artifact_path="manifests")

        logger.info(f"stage 02 complete: manifests at {output_dir}")


if __name__ == "__main__":
    run()
