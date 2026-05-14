from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger


_INVALID_LABEL_RE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_label_value(raw: str) -> str:
    """Sanitize a string for use as a Prometheus label value.

    Replaces characters outside ``[a-zA-Z0-9_]`` with ``_`` and prepends
    ``_`` if the result starts with a digit.
    """
    sanitized = _INVALID_LABEL_RE.sub("_", raw)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_empty"


TRAINING_RUNS_TOTAL = Counter(
    "retinaxai_training_runs_total",
    "Total number of training runs triggered",
    ["pipeline"],
)

TRAINING_FAILURES_TOTAL = Counter(
    "retinaxai_training_failures_total",
    "Total number of training job failures",
    ["pipeline", "error_type"],
)

ACTIVE_TRAINING_JOBS = Gauge(
    "retinaxai_active_training_jobs",
    "Number of currently running training jobs",
)

BEST_VAL_ACCURACY = Gauge(
    "retinaxai_best_val_accuracy",
    "Best validation accuracy from last training run",
    ["pipeline"],
)

QUADRATIC_WEIGHTED_KAPPA = Gauge(
    "retinaxai_quadratic_weighted_kappa",
    "Quadratic weighted kappa from last evaluation run",
    ["pipeline", "split"],
)

EPOCH_TRAIN_LOSS = Histogram(
    "retinaxai_epoch_train_loss",
    "Training loss per epoch",
    ["pipeline"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)

INFERENCE_LATENCY = Histogram(
    "retinaxai_inference_latency_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

INFERENCE_OOM_KILLS = Counter(
    "retinaxai_inference_oom_kills",
    "Total number of CUDA OOM recoveries during inference",
    ["model"],
)

PREDICTION_REQUESTS_TOTAL = Counter(
    "retinaxai_prediction_requests_total",
    "Total number of prediction requests",
    ["model"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "retinaxai_prediction_errors_total",
    "Total number of prediction errors",
    ["model", "error_type"],
)

GRADCAM_GENERATION_FAILURES = Counter(
    "retinaxai_gradcam_generation_failures",
    "Total number of GradCAM generation failures",
    ["eye"],
)

GPU_MEMORY_USED_BYTES = Gauge(
    "retinaxai_gpu_memory_used_bytes",
    "GPU memory currently used in bytes",
    ["device"],
)

GPU_UTILIZATION_PERCENT = Gauge(
    "retinaxai_gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
)

DRIFT_DETECTED = Gauge(
    "retinaxai_drift_detected",
    "Whether drift was detected (1) or not (0)",
    ["pipeline"],
)

DRIFT_PSI_SCORE = Gauge(
    "retinaxai_drift_psi_score",
    "Population Stability Index (PSI) for drift detection",
    ["pipeline", "feature"],
)

EVIDENTLY_DRIFT_DATASET_SHIFT = Gauge(
    "retinaxai_evidently_dataset_shift",
    "Evidently dataset drift score (0-1)",
    ["pipeline"],
)

EVIDENTLY_DRIFT_FEATURES_DRIFTED = Gauge(
    "retinaxai_evidently_features_drifted",
    "Number of features showing drift per Evidently",
    ["pipeline"],
)

AUTOMATION_SCHEDULER_RUNNING = Gauge(
    "retinaxai_automation_scheduler_running",
    "Whether automation scheduler is running (1) or not (0)",
)

TRAINING_REJECTIONS_TOTAL = Counter(
    "retinaxai_training_rejections_total",
    "Total number of training job rejections due to capacity limits",
    ["pipeline", "reason"],
)

TRAINING_SLOTS_USED = Gauge(
    "retinaxai_training_slots_used",
    "Number of training slots currently used",
    ["pipeline"],
)

MODEL_REGISTRY_VERSIONS = Gauge(
    "retinaxai_model_registry_versions",
    "Number of model versions in registry",
    ["pipeline", "stage"],
)

MODEL_PROMOTIONS_TOTAL = Counter(
    "retinaxai_model_promotions_total",
    "Total number of model promotions",
    ["pipeline"],
)

MODEL_ROLLBACKS_TOTAL = Counter(
    "retinaxai_model_rollbacks_total",
    "Total number of model rollbacks",
    ["pipeline"],
)

TRAINING_CURRENT_EPOCH = Gauge(
    "retinaxai_training_current_epoch",
    "Current training epoch number",
    ["pipeline"],
)

TRAINING_TOTAL_EPOCHS = Gauge(
    "retinaxai_training_total_epochs",
    "Total number of epochs for this training run",
    ["pipeline"],
)

TRAINING_EPOCH_ACCURACY = Gauge(
    "retinaxai_training_epoch_accuracy",
    "Accuracy per epoch for train and validation splits",
    ["pipeline", "split"],
)

TRAINING_EPOCH_F1 = Gauge(
    "retinaxai_training_epoch_f1",
    "Macro F1 score per epoch for train and validation splits",
    ["pipeline", "split"],
)

TRAINING_LEARNING_RATE = Gauge(
    "retinaxai_training_learning_rate",
    "Current learning rate during training",
    ["pipeline"],
)

TRAINING_EPOCH_DURATION = Gauge(
    "retinaxai_training_epoch_duration_seconds",
    "Duration of each training epoch in seconds",
    ["pipeline"],
)

TRAINING_EPOCH_PSI = Gauge(
    "retinaxai_training_epoch_psi",
    "Population Stability Index between train and val predictions per epoch",
    ["pipeline"],
)

TRAINING_BEST_F1 = Gauge(
    "retinaxai_training_best_f1",
    "Best macro F1 score achieved during training",
    ["pipeline"],
)

TRAINING_PATIENCE_COUNTER = Gauge(
    "retinaxai_training_patience_counter",
    "Current early stopping patience counter",
    ["pipeline"],
)

TRAINING_VAL_LOSS = Gauge(
    "retinaxai_training_val_loss",
    "Validation loss per epoch",
    ["pipeline"],
)

TRAINING_PER_CLASS_F1 = Gauge(
    "retinaxai_training_per_class_f1",
    "Per-class F1 score during training",
    ["pipeline", "dr_grade"],
)

EVALUATION_SCORE = Gauge(
    "retinaxai_evaluation_score",
    "Evaluation metrics from last pipeline run (accuracy, roc_auc, F1, precision, recall)",
    ["pipeline", "split", "metric"],
)


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = Σ((actual% - expected%) × ln(actual% / expected%))

    Thresholds:
        < 0.1: No significant change
        0.1 - 0.25: Moderate change
        > 0.25: Significant change requiring investigation
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    eps = 1e-6
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, buckets + 1)

    expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected) + eps
    actual_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


_METRIC_KEYS = [
    "accuracy",
    "roc_auc_macro",
    "macro_f1",
    "precision_macro",
    "recall_macro",
]


def init_metrics() -> None:
    """Initialize all Gauges to 0 so Prometheus has data to scrape at startup."""
    p = "imaging"
    DRIFT_DETECTED.labels(pipeline=p).set(0)
    EVIDENTLY_DRIFT_DATASET_SHIFT.labels(pipeline=p).set(0)
    EVIDENTLY_DRIFT_FEATURES_DRIFTED.labels(pipeline=p).set(0)
    BEST_VAL_ACCURACY.labels(pipeline=p).set(0)
    TRAINING_SLOTS_USED.labels(pipeline=p).set(0)
    MODEL_REGISTRY_VERSIONS.labels(pipeline=p, stage="staging").set(0)
    MODEL_REGISTRY_VERSIONS.labels(pipeline=p, stage="production").set(0)
    TRAINING_CURRENT_EPOCH.labels(pipeline=p).set(0)
    TRAINING_TOTAL_EPOCHS.labels(pipeline=p).set(0)
    TRAINING_EPOCH_ACCURACY.labels(pipeline=p, split="train").set(0)
    TRAINING_EPOCH_ACCURACY.labels(pipeline=p, split="val").set(0)
    TRAINING_EPOCH_F1.labels(pipeline=p, split="train").set(0)
    TRAINING_EPOCH_F1.labels(pipeline=p, split="val").set(0)
    TRAINING_LEARNING_RATE.labels(pipeline=p).set(0)
    TRAINING_EPOCH_DURATION.labels(pipeline=p).set(0)
    TRAINING_EPOCH_PSI.labels(pipeline=p).set(0)
    TRAINING_BEST_F1.labels(pipeline=p).set(0)
    TRAINING_PATIENCE_COUNTER.labels(pipeline=p).set(0)
    TRAINING_VAL_LOSS.labels(pipeline=p).set(0)
    for grade in range(5):
        TRAINING_PER_CLASS_F1.labels(pipeline=p, dr_grade=str(grade)).set(0)
    for split in ("eyepacs_test", "samaya_validation"):
        for metric in _METRIC_KEYS:
            EVALUATION_SCORE.labels(pipeline="imaging", split=split, metric=metric).set(
                0
            )
    ACTIVE_TRAINING_JOBS.set(0)
    AUTOMATION_SCHEDULER_RUNNING.set(0)
    TRAINING_SLOTS_USED.labels(pipeline="all").set(0)
    GPU_MEMORY_USED_BYTES.labels(device="0").set(0)
    GPU_UTILIZATION_PERCENT.labels(device="0").set(0)
    logger.info("prometheus metrics initialized")


def start_metrics_server(port: int = 9101) -> None:
    try:
        start_http_server(port)
        logger.info(f"prometheus metrics server started on port {port}")
    except OSError as e:
        logger.warning(f"Could not start metrics server on port {port}: {e}")


def update_qwk_from_metrics_files(
    imaging_metrics_path: Path,
) -> None:
    """Read persisted metrics.json file and update QWK gauges.

    The evaluation pipeline writes metrics.json to disk, but runs as a
    transient CLI process (python main.py pipeline) that exits before
    Prometheus can scrape.  This bridge lets the long-lived FastAPI
    process pick up the persisted values so the dashboard has data.
    """
    if imaging_metrics_path.exists():
        try:
            with open(imaging_metrics_path) as f:
                data = json.load(f)
            eyepacs = data.get("eyepacs_test", {})
            if "quadratic_weighted_kappa" in eyepacs:
                QUADRATIC_WEIGHTED_KAPPA.labels(
                    pipeline="imaging", split="eyepacs_test"
                ).set(eyepacs["quadratic_weighted_kappa"])
            samaya = data.get("samaya_validation", {}) or {}
            if "quadratic_weighted_kappa" in samaya:
                QUADRATIC_WEIGHTED_KAPPA.labels(
                    pipeline="imaging", split="samaya_validation"
                ).set(samaya["quadratic_weighted_kappa"])
        except Exception as e:
            logger.warning(f"failed to load imaging QWK: {e}")


def start_qwk_background_refresh(
    imaging_metrics_path: Path,
    interval_seconds: int = 300,
) -> None:
    """Start a daemon thread that periodically reloads QWK from metrics files."""

    def _refresh_loop() -> None:
        while True:
            try:
                update_qwk_from_metrics_files(imaging_metrics_path)
            except Exception as e:
                logger.warning(f"QWK background refresh error: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_refresh_loop, daemon=True)
    thread.start()
    logger.info(f"QWK background refresh started (interval={interval_seconds}s)")


def update_drift_metrics_from_files(
    drift_history_path: Path,
    evidently_metrics_path: Path,
) -> None:
    """Read persisted drift-check results and update Prometheus gauges.

    The pipeline and automation service both write drift results to disk;
    this bridge makes them visible to Prometheus scrapes in the FastAPI
    process, just like the QWK bridge.
    """
    pipelines = ["imaging"]

    # --- PSI drift from drift_history.json ---
    if drift_history_path.exists():
        try:
            with open(drift_history_path) as f:
                history: list[dict] = json.load(f)

            for pipe in pipelines:
                pipe_entries = [h for h in history if h.get("pipeline") == pipe]
                if not pipe_entries:
                    continue
                latest = pipe_entries[-1]
                DRIFT_DETECTED.labels(pipeline=pipe).set(
                    1 if latest.get("drift_detected", False) else 0
                )
                overall_psi = latest.get("overall_psi", 0.0)
                DRIFT_PSI_SCORE.labels(pipeline=pipe, feature="_overall").set(
                    overall_psi
                )
                for feature in latest.get("features", []):
                    raw_name = feature.get("feature_name", "")
                    safe_name = sanitize_label_value(str(raw_name))
                    try:
                        DRIFT_PSI_SCORE.labels(pipeline=pipe, feature=safe_name).set(
                            feature.get("psi", 0.0)
                        )
                    except Exception:
                        logger.debug(
                            "skipping invalid PSI feature label",
                            feature=raw_name,
                        )
        except Exception as e:
            logger.warning(f"failed to load drift PSI from history: {e}")

    # --- Evidently metrics from evidently_metrics.json ---
    if evidently_metrics_path.exists():
        try:
            with open(evidently_metrics_path) as f:
                metrics: dict = json.load(f)
            for pipe in pipelines:
                entry = metrics.get(pipe, {})
                dataset_drift = entry.get("dataset_drift", 0.0)
                features_drifted = entry.get("features_drifted", 0)
                EVIDENTLY_DRIFT_DATASET_SHIFT.labels(pipeline=pipe).set(dataset_drift)
                EVIDENTLY_DRIFT_FEATURES_DRIFTED.labels(pipeline=pipe).set(
                    features_drifted
                )
        except Exception as e:
            logger.warning(f"failed to load evidently metrics: {e}")


def start_drift_background_refresh(
    drift_history_path: Path,
    evidently_metrics_path: Path,
    interval_seconds: int = 300,
) -> None:
    """Start a daemon thread that periodically reloads drift gauges."""

    def _refresh_loop() -> None:
        while True:
            try:
                update_drift_metrics_from_files(
                    drift_history_path, evidently_metrics_path
                )
            except Exception as e:
                logger.warning(f"drift background refresh error: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_refresh_loop, daemon=True)
    thread.start()
    logger.info(f"drift background refresh started (interval={interval_seconds}s)")


def update_evaluation_metrics_from_files(imaging_metrics_path: Path) -> None:
    """Read persisted metrics.json and set evaluation score gauges.

    Bridges evaluation metrics (accuracy, ROC-AUC, F1, precision, recall)
    from the transient CLI pipeline into Prometheus via the same file
    mechanism as the QWK bridge.
    """
    if not imaging_metrics_path.exists():
        return
    try:
        with open(imaging_metrics_path) as f:
            data = json.load(f)
        for split in ("eyepacs_test", "samaya_validation"):
            split_data = data.get(split) or {}
            for metric in _METRIC_KEYS:
                if metric in split_data:
                    EVALUATION_SCORE.labels(
                        pipeline="imaging", split=split, metric=metric
                    ).set(split_data[metric])
    except Exception as e:
        logger.warning(f"failed to load evaluation metrics: {e}")


def start_evaluation_background_refresh(
    imaging_metrics_path: Path,
    interval_seconds: int = 300,
) -> None:
    """Start a daemon thread that periodically reloads evaluation score gauges."""

    def _refresh_loop() -> None:
        while True:
            try:
                update_evaluation_metrics_from_files(imaging_metrics_path)
            except Exception as e:
                logger.warning(f"evaluation background refresh error: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_refresh_loop, daemon=True)
    thread.start()
    logger.info(f"evaluation background refresh started (interval={interval_seconds}s)")
