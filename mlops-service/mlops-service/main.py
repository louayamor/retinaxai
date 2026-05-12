import os
from pathlib import Path
from collections.abc import Callable

import argparse
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
os.chdir(Path(__file__).parent)

from app.domains.imaging.pipeline.stage_01_data_ingestion import run as img_ingest  # noqa: E402
from app.domains.imaging.pipeline.stage_02_data_cleaning import run as img_clean  # noqa: E402
from app.domains.imaging.pipeline.stage_03_data_transformation import (
    run as img_transform,
)  # noqa: E402
from app.domains.imaging.pipeline.stage_04_model_trainer import run as img_train  # noqa: E402
from app.domains.imaging.pipeline.stage_05_model_evaluation import run as img_evaluate  # noqa: E402

from app.domains.clinical.pipeline.stage_01_data_ingestion import run as clin_ingest  # noqa: E402
from app.domains.clinical.pipeline.stage_02_data_cleaning import run as clin_clean  # noqa: E402
from app.domains.clinical.pipeline.stage_03_data_transformation import (
    run as clin_transform,
)  # noqa: E402
from app.domains.clinical.pipeline.stage_04_model_trainer import run as clin_train  # noqa: E402
from app.domains.clinical.pipeline.stage_05_model_evaluation import run as clin_evaluate  # noqa: E402

IMAGING_PIPELINE: dict[str, Callable] = {
    "ingest": img_ingest,
    "clean": img_clean,
    "transform": img_transform,
    "train": img_train,
    "evaluate": img_evaluate,
}

CLINICAL_PIPELINE: dict[str, Callable] = {
    "ingest": clin_ingest,
    "clean": clin_clean,
    "transform": clin_transform,
    "train": clin_train,
    "evaluate": clin_evaluate,
}

PIPELINE_ORDER = ["ingest", "clean", "transform", "train", "evaluate"]

from app.utils.mlflow_utils import configure_mlflow  # noqa: E402
from app.services.monitoring.prometheus_metrics import (
    TRAINING_RUNS_TOTAL,
    start_metrics_server,
)  # noqa: E402


def run_stage(stage: str, pipeline: dict[str, Callable]) -> None:
    if stage not in pipeline:
        raise ValueError(f"Invalid stage: {stage}")
    logger.info(f"Running stage: {stage}")
    pipeline[stage]()


def run_full_pipeline(pipeline: dict[str, Callable]) -> None:
    for stage in PIPELINE_ORDER:
        run_stage(stage, pipeline)


def run_pipeline(stage: str, target: str) -> None:
    start_metrics_server()

    if stage in ("train", "evaluate", "all"):
        configure_mlflow()

    targets = []
    if target in ("imaging", "both"):
        targets.append(("imaging", IMAGING_PIPELINE))
    if target in ("clinical", "both"):
        targets.append(("clinical", CLINICAL_PIPELINE))

    for name, pipe in targets:
        TRAINING_RUNS_TOTAL.labels(pipeline=name).inc()
        logger.info(f"Executing {name} pipeline")

        if stage == "all":
            run_full_pipeline(pipe)
        else:
            run_stage(stage, pipe)


def serve() -> None:
    import uvicorn
    from app.config.settings import Settings

    settings = Settings()

    logger.info(f"Starting API server at {settings.app_host}:{settings.app_port}")

    uvicorn.run(
        "app.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


def main():
    parser = argparse.ArgumentParser(description="RetinaXAI MLOps Service")

    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument(
        "--stage",
        type=str,
        choices=["ingest", "clean", "transform", "train", "evaluate", "all"],
        default="all",
    )
    pipeline_parser.add_argument(
        "--pipeline",
        type=str,
        choices=["imaging", "clinical", "both"],
        default="both",
        help="Pipeline to run (imaging now uses 2-phase training by default)",
    )

    subparsers.add_parser("serve")

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline(args.stage, args.pipeline)

    elif args.command == "serve":
        serve()


if __name__ == "__main__":
    main()
