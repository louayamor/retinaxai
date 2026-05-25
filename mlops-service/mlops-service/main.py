import os
from pathlib import Path
from collections.abc import Callable

import argparse
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
os.chdir(Path(__file__).parent)

from app.training.pipeline.stage_01_data_ingestion import run as img_ingest  # noqa: E402
from app.training.pipeline.stage_02_data_cleaning import run as img_clean  # noqa: E402
from app.training.pipeline.stage_03_data_transformation import (
    run as img_transform,
)  # noqa: E402
from app.training.pipeline.stage_04_model_trainer import run as img_train  # noqa: E402
from app.training.pipeline.stage_04b_fundus_classifier import (
    run as img_fundus,
)  # noqa: E402
from app.training.pipeline.stage_05_model_evaluation import run as img_evaluate  # noqa: E402

from app.training.pipeline.lesion.stage_01_ddr_ingestion import (  # noqa: E402
    run as lesion_ingest,
)
from app.training.pipeline.lesion.stage_02_ddr_transformation import (  # noqa: E402
    run as lesion_transform,
)
from app.training.pipeline.lesion.stage_03_lesion_training import (  # noqa: E402
    run as lesion_train,
)
from app.training.pipeline.lesion.stage_04_lesion_evaluation import (  # noqa: E402
    run as lesion_evaluate,
)

IMAGING_PIPELINE: dict[str, Callable] = {
    "ingest": img_ingest,
    "clean": img_clean,
    "transform": img_transform,
    "fundus": img_fundus,
    "train": img_train,
    "evaluate": img_evaluate,
}

IMAGING_PIPELINE_ORDER = ["ingest", "clean", "transform", "fundus", "train", "evaluate"]

LESION_PIPELINE: dict[str, Callable] = {
    "ingest": lesion_ingest,
    "transform": lesion_transform,
    "train": lesion_train,
    "evaluate": lesion_evaluate,
}

LESION_PIPELINE_ORDER = ["ingest", "transform", "train", "evaluate"]

from app.utils.mlflow_utils import configure_mlflow  # noqa: E402
from app.monitoring.prometheus_metrics import (
    TRAINING_RUNS_TOTAL,
    start_metrics_server,
)  # noqa: E402


def run_stage(stage: str, pipeline: dict[str, Callable]) -> None:
    if stage not in pipeline:
        raise ValueError(f"Invalid stage: {stage}")
    logger.info(f"Running stage: {stage}")
    pipeline[stage]()


def run_full_pipeline(
    pipeline: dict[str, Callable], order: list[str]
) -> None:
    for stage in order:
        run_stage(stage, pipeline)


def run_pipeline(stage: str, target: str) -> None:
    start_metrics_server()

    if stage in ("train", "evaluate", "all"):
        configure_mlflow()

    if target == "lesion":
        TRAINING_RUNS_TOTAL.labels(pipeline="lesion").inc()
        logger.info("Executing lesion pipeline")
        if stage == "all":
            run_full_pipeline(LESION_PIPELINE, LESION_PIPELINE_ORDER)
        else:
            run_stage(stage, LESION_PIPELINE)
    else:
        TRAINING_RUNS_TOTAL.labels(pipeline="imaging").inc()
        logger.info("Executing imaging pipeline")
        if stage == "all":
            run_full_pipeline(IMAGING_PIPELINE, IMAGING_PIPELINE_ORDER)
        else:
            run_stage(stage, IMAGING_PIPELINE)


def serve() -> None:
    import sys

    import uvicorn
    from app.config.settings import Settings

    # Stage imports above replace loguru handler with JSON-serialized format;
    # reset to human-readable for the API server.
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} - {message}",
    )

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
        choices=["ingest", "clean", "transform", "fundus", "train", "evaluate", "all"],
        default="all",
    )
    pipeline_parser.add_argument(
        "--pipeline",
        type=str,
        choices=["imaging", "lesion"],
        default="imaging",
        help="Pipeline to run (imaging: DR grading; lesion: lesion segmentation)",
    )

    subparsers.add_parser("serve")

    args = parser.parse_args()

    if args.command == "pipeline":
        run_pipeline(args.stage, args.pipeline)

    elif args.command == "serve":
        serve()


if __name__ == "__main__":
    main()
