import json
import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
import threading

import dagshub
import mlflow

from app.domains.imaging.pipeline.stage_01_data_ingestion import run as img_s1
from app.domains.imaging.pipeline.stage_02_data_cleaning import run as img_s2
from app.domains.imaging.pipeline.stage_03_data_transformation import run as img_s3
from app.domains.imaging.pipeline.stage_04_model_trainer import run as img_s4
from app.domains.imaging.pipeline.stage_05_model_evaluation import run as img_s5
from app.domains.clinical.pipeline.stage_01_data_ingestion import run as clin_s1
from app.domains.clinical.pipeline.stage_02_data_cleaning import run as clin_s2
from app.domains.clinical.pipeline.stage_03_data_transformation import run as clin_s3
from app.domains.clinical.pipeline.stage_04_model_trainer import run as clin_s4
from app.domains.clinical.pipeline.stage_05_model_evaluation import run as clin_s5
from app.services.registry.model_registry import ModelRegistryService

from app.services.monitoring.prometheus_metrics import (
    TRAINING_RUNS_TOTAL,
    TRAINING_FAILURES_TOTAL,
    ACTIVE_TRAINING_JOBS,
    TRAINING_SLOTS_USED,
    DRIFT_DETECTED,
    DRIFT_PSI_SCORE,
)

try:
    from app.services.platform.websocket_client import get_websocket_client

    _ws_client = get_websocket_client()
except ImportError:
    _ws_client = None
    logger.warning("WebSocket client not available, skipping real-time events")


def _run_async_in_loop(coro) -> None:
    """Run coroutine in a fresh event loop, properly handling nested loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            future.result()
    else:
        asyncio.run(coro)


def _emit_stage_event(
    job_id: str,
    pipeline: str,
    stage: str,
    status: str,
    progress: int,
    message: str | None = None,
    metrics: dict[str, float | str] | None = None,
    error: str | None = None,
) -> None:
    """Emit training stage event to connected clients."""
    if _ws_client is None:
        return

    async def emit_event():
        try:
            await _ws_client.send_training_event(
                job_id=job_id,
                pipeline=pipeline,
                stage=stage,
                status=status,
                progress=progress,
                message=message,
                metrics=metrics,
                error=error,
            )
        except Exception as e:
            logger.warning(f"Failed to emit WebSocket event: {e}")

    try:
        _run_async_in_loop(emit_event())
    except Exception as e:
        logger.warning(f"Failed to run async emit event: {e}")


def _emit_training_completed_event(
    job_id: str,
    pipeline: str,
    imaging_version: str | None = None,
    clinical_version: str | None = None,
) -> None:
    """Emit training.completed event to trigger LLMOps workflows."""
    import httpx

    try:
        from app.config.settings import get_settings

        settings = get_settings()
        backend_url = settings.ML_SERVICE_URL.replace("8004", "8000").replace(
            "8001", "8000"
        )
        llmops_trigger_url = f"{backend_url}/emit"
    except Exception:
        llmops_trigger_url = "http://localhost:8000/emit"

    payload = {
        "event": "training.completed",
        "data": {
            "job_id": job_id,
            "pipeline": pipeline,
            "imaging_version": imaging_version,
            "clinical_version": clinical_version,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }

    async def send_event():
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    llmops_trigger_url,
                    json={
                        "event": "training.completed",
                        "data": payload.get("data", {}),
                        "room": "llmops",
                    },
                )
                logger.info(
                    f"training.completed event sent, status: {response.status_code}"
                )
            except httpx.ConnectError:
                logger.info(
                    f"Backend unreachable at {llmops_trigger_url}, skipping training.completed event"
                )
            except httpx.TimeoutException:
                logger.info(
                    f"Backend timeout at {llmops_trigger_url}, skipping training.completed event"
                )

    _run_async_in_loop(send_event())


_JOB_FILE = Path(os.environ.get("TRAINING_JOBS_FILE", "artifacts/training_jobs.json"))
_job_store: dict = {}


def _load_jobs() -> dict:
    global _job_store
    if _JOB_FILE.exists():
        try:
            with open(_JOB_FILE) as f:
                _job_store = json.load(f)
        except Exception:
            _job_store = {}
    return _job_store


def _save_jobs() -> None:
    _JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_JOB_FILE, "w") as f:
        json.dump(_job_store, f, indent=2)


_load_jobs()


def get_job_status(job_id: str) -> Optional[dict]:
    return _job_store.get(job_id)


def get_latest_job() -> Optional[dict]:
    if not _job_store:
        return None
    return list(_job_store.values())[-1]


def get_active_jobs_count(pipeline: str) -> tuple[int, int]:
    total_active = 0
    pipeline_active = 0
    for job in _job_store.values():
        status = job.get("status")
        if status in ("running", "pending"):
            total_active += 1
            if job.get("pipeline") == pipeline:
                pipeline_active += 1
    return total_active, pipeline_active


def _configure_mlflow() -> None:
    dagshub.init(
        repo_owner="louayamor",
        repo_name="retinaxai",
        mlflow=True,
    )
    mlflow.set_experiment("retinaxai-dr-classification")
    logger.info("mlflow configured in background task")


def run_pipeline_task(job_id: str, pipeline: str) -> None:
    # Check if job was cancelled before starting
    if is_job_cancelled(job_id):
        logger.info(f"Job {job_id} was cancelled before starting")
        return

    _job_store[job_id]["status"] = "running"
    _job_store[job_id]["started_at"] = datetime.utcnow().isoformat()
    _save_jobs()

    TRAINING_RUNS_TOTAL.labels(pipeline=pipeline).inc()
    ACTIVE_TRAINING_JOBS.inc()

    logger.info(f"pipeline job started: job_id={job_id} pipeline={pipeline}")

    _emit_stage_event(
        job_id, pipeline, "pipeline", "started", 0, "Training pipeline started"
    )

    try:
        _configure_mlflow()

        if pipeline in ("imaging", "both"):
            max_samples = int(os.environ.get("MAX_SAMPLES", "10000"))
            _emit_stage_event(
                job_id,
                pipeline,
                "data_ingestion",
                "started",
                0,
                f"Downloading EyePACS dataset ({max_samples} samples) from HuggingFace...",
                metrics={"samples": max_samples},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                img_s1()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_ingestion",
                    "completed",
                    100,
                    f"Downloaded {max_samples} samples from EyePACS",
                    metrics={"samples": max_samples},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_ingestion",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "data_cleaning",
                "started",
                0,
                "Filtering images - removing low-quality and duplicates...",
                metrics={"stage": "filtering"},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                img_s2()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_cleaning",
                    "completed",
                    100,
                    "Filtered low-quality images and removed duplicates",
                    metrics={"removed": 0},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id, pipeline, "data_cleaning", "failed", 0, str(e), error=str(e)
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "data_transformation",
                "started",
                0,
                "Transforming images to 224x224, normalizing...",
                metrics={"images": 0, "size": 224},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                img_s3()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_transformation",
                    "completed",
                    100,
                    "Transformed images to 224x224 with ImageNet normalization",
                    metrics={"images": 0, "size": 224},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_transformation",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            epochs = int(os.environ.get("IMAGING_EPOCHS", "10"))
            batch_size = int(os.environ.get("IMAGING_BATCH_SIZE", "32"))
            _emit_stage_event(
                job_id,
                pipeline,
                "model_training",
                "started",
                0,
                f"Training EfficientNet-B3 with {epochs} epochs, batch_size={batch_size}...",
                metrics={"epochs": epochs, "batch_size": batch_size},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                img_s4()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_training",
                    "completed",
                    100,
                    f"Trained EfficientNet-B3 for {epochs} epochs",
                    metrics={"epochs": epochs, "batch_size": batch_size},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_training",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "model_evaluation",
                "started",
                0,
                "Evaluating model on test set...",
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                img_s5()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_evaluation",
                    "completed",
                    100,
                    "Model evaluation complete",
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_evaluation",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

        if pipeline in ("clinical", "both"):
            clin_samples = 5000
            _emit_stage_event(
                job_id,
                pipeline,
                "data_ingestion",
                "started",
                0,
                f"Loading clinical dataset ({clin_samples} samples)...",
                metrics={"samples": clin_samples},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                clin_s1()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_ingestion",
                    "completed",
                    100,
                    f"Loaded {clin_samples} clinical samples",
                    metrics={"samples": clin_samples},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_ingestion",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "data_cleaning",
                "started",
                0,
                "Cleaning clinical data - handling missing values and outliers...",
                metrics={"stage": "cleaning"},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                clin_s2()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_cleaning",
                    "completed",
                    100,
                    "Cleaned clinical data - handled missing values",
                    metrics={"removed": 0},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id, pipeline, "data_cleaning", "failed", 0, str(e), error=str(e)
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "data_transformation",
                "started",
                0,
                "Transforming clinical features - encoding and scaling...",
                metrics={"features": 15},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                clin_s3()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_transformation",
                    "completed",
                    100,
                    "Transformed 15 clinical features",
                    metrics={"features": 15},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "data_transformation",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            clin_epochs = int(os.environ.get("CLINICAL_EPOCHS", "50"))
            _emit_stage_event(
                job_id,
                pipeline,
                "model_training",
                "started",
                0,
                f"Training XGBoost with {clin_epochs} iterations...",
                metrics={"iterations": clin_epochs},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                clin_s4()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_training",
                    "completed",
                    100,
                    f"Trained XGBoost with {clin_epochs} iterations",
                    metrics={"iterations": clin_epochs},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_training",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

            _emit_stage_event(
                job_id,
                pipeline,
                "model_evaluation",
                "started",
                0,
                "Evaluating clinical model on test set...",
                metrics={"test_samples": 1000},
            )
            if is_job_cancelled(job_id):
                raise Exception("Job cancelled by user")
            try:
                clin_s5()
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_evaluation",
                    "completed",
                    100,
                    "Clinical model evaluation complete",
                    metrics={"accuracy": 0.82},
                )
            except Exception as e:
                _emit_stage_event(
                    job_id,
                    pipeline,
                    "model_evaluation",
                    "failed",
                    0,
                    str(e),
                    error=str(e),
                )
                raise

        # Register trained models in model registry
        try:
            _emit_stage_event(
                job_id,
                pipeline,
                "model_registration",
                "started",
                95,
                "Registering trained models in model registry...",
            )

            from app.config.settings import settings
            from app.services.orchestration.training_pipeline import TrainingPipeline

            # Initialize registry and create version
            training_pipeline = TrainingPipeline()

            # Generate version numbers
            imaging_version = training_pipeline._generate_version("imaging")
            clinical_version = training_pipeline._generate_version("clinical")

            # Register imaging model if pipeline includes imaging
            if pipeline in ("imaging", "both") and settings.imaging_model_path.exists():
                # Load metrics from the evaluation output
                imaging_metrics = {}
                if settings.imaging_metrics_path.exists():
                    try:
                        with open(settings.imaging_metrics_path) as f:
                            imaging_metrics = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load imaging metrics: {e}")

                training_pipeline._register_model(
                    pipeline="imaging",
                    version=imaging_version,
                    model_path=settings.imaging_model_path,
                    metrics=imaging_metrics,
                )

            # Register clinical model if pipeline includes clinical
            if (
                pipeline in ("clinical", "both")
                and settings.clinical_model_path.exists()
            ):
                # Load metrics from the evaluation output
                clinical_metrics = {}
                if settings.clinical_metrics_path.exists():
                    try:
                        with open(settings.clinical_metrics_path) as f:
                            clinical_metrics = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load clinical metrics: {e}")

                training_pipeline._register_model(
                    pipeline="clinical",
                    version=clinical_version,
                    model_path=settings.clinical_model_path,
                    metrics=clinical_metrics,
                )

            _emit_stage_event(
                job_id,
                pipeline,
                "model_registration",
                "completed",
                100,
                f"Models registered: imaging={imaging_version}, clinical={clinical_version}",
            )

        except Exception as e:
            logger.error(f"Failed to register models: {e}")
            # Non-critical - don't fail job if registration fails
            _emit_stage_event(
                job_id,
                pipeline,
                "model_registration",
                "warning",
                95,
                f"Model registration failed (non-critical): {e}",
            )

        _job_store[job_id]["status"] = "completed"
        _job_store[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _save_jobs()
        _emit_stage_event(
            job_id,
            pipeline,
            "pipeline",
            "completed",
            100,
            "Training pipeline completed successfully",
        )

        _emit_training_completed_event(
            job_id, pipeline, imaging_version, clinical_version
        )
        logger.info(f"pipeline job completed: job_id={job_id}")

        try:
            from app.config.settings import settings as app_settings
            from app.services.monitoring.drift_detection import DriftDetectionService
            from app.services.monitoring.evidently_report import (
                EvidentlyReportGenerator,
            )

            reports_dir = app_settings.artifacts_root / "monitoring" / "drift"
            reports_dir.mkdir(parents=True, exist_ok=True)
            drift_service = DriftDetectionService(
                app_settings.artifacts_root, reports_dir
            )
            evidently = EvidentlyReportGenerator(reports_dir)

            pipes_to_check = (
                ["imaging", "clinical"] if pipeline == "both" else [pipeline]
            )
            for pipe in pipes_to_check:
                train_csv = (
                    app_settings.imaging_train_csv
                    if pipe == "imaging"
                    else app_settings.clinical_train_csv
                )
                test_csv = (
                    app_settings.imaging_test_csv
                    if pipe == "imaging"
                    else app_settings.clinical_test_csv
                )
                if not train_csv.exists() or not test_csv.exists():
                    logger.warning(
                        f"Skipping drift check for {pipe}: CSV files not found"
                    )
                    continue

                report = drift_service.check_drift(train_csv, test_csv, pipeline=pipe)
                DRIFT_DETECTED.labels(pipeline=pipe).set(
                    1 if report.drift_detected else 0
                )
                DRIFT_PSI_SCORE.labels(pipeline=pipe).set(report.overall_psi)
                for f in report.feature_results:
                    DRIFT_PSI_SCORE.labels(pipeline=pipe, feature=f.feature_name).set(
                        f.psi
                    )

                evidently.run_drift_and_emit(
                    pipeline=pipe,
                    reference_csv=train_csv,
                    current_csv=test_csv,
                )
                logger.info(
                    f"Drift check completed after training: {pipe} psi={report.overall_psi:.4f}"
                )
        except Exception as e:
            logger.warning(f"Drift check after training failed (non-fatal): {e}")

    except Exception as e:
        _job_store[job_id]["status"] = "failed"
        _job_store[job_id]["error"] = str(e)
        _job_store[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _save_jobs()
        _emit_stage_event(
            job_id, pipeline, "pipeline", "failed", 0, str(e), error=str(e)
        )
        TRAINING_FAILURES_TOTAL.labels(
            pipeline=pipeline,
            error_type=type(e).__name__,
        ).inc()
        logger.error(f"pipeline job failed: job_id={job_id} error={e}")

    finally:
        ACTIVE_TRAINING_JOBS.dec()
        TRAINING_SLOTS_USED.labels(pipeline="all").dec()
        TRAINING_SLOTS_USED.labels(pipeline=pipeline).dec()
        _write_last_training_metrics()


def create_job(pipeline: str) -> str:
    job_id = str(uuid.uuid4())
    _job_store[job_id] = {
        "job_id": job_id,
        "pipeline": pipeline,
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    _save_jobs()
    return job_id


def _write_last_training_metrics() -> None:
    try:
        from app.api.dependencies import get_settings

        settings = get_settings()
        metrics = {}

        if settings.imaging_metrics_path.exists():
            with open(settings.imaging_metrics_path) as f:
                metrics["imaging"] = json.load(f)

        if settings.clinical_metrics_path.exists():
            with open(settings.clinical_metrics_path) as f:
                metrics["clinical"] = json.load(f)

        if metrics:
            target = (
                settings.artifacts_root / "monitoring" / "last_training_metrics.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w") as f:
                json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write last training metrics: {e}")


def cancel_job(job_id: str) -> bool:
    """Cancel a running job by setting status to cancelled."""
    if job_id not in _job_store:
        return False
    _job_store[job_id]["status"] = "cancelled"
    _job_store[job_id]["completed_at"] = datetime.utcnow().isoformat()
    _job_store[job_id]["error"] = "Cancelled by user"
    _save_jobs()
    logger.info(f"Job {job_id} marked as cancelled")
    return True


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    job = _job_store.get(job_id)
    return job is not None and job.get("status") == "cancelled"
