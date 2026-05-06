from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger


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
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
)

INFERENCE_LATENCY = Histogram(
    "retinaxai_inference_latency_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
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


def init_metrics() -> None:
    """Initialize all Gauges to 0 so Prometheus has data to scrape at startup."""
    pipelines = ["imaging", "clinical"]
    for p in pipelines:
        DRIFT_DETECTED.labels(pipeline=p).set(0)
        EVIDENTLY_DRIFT_DATASET_SHIFT.labels(pipeline=p).set(0)
        EVIDENTLY_DRIFT_FEATURES_DRIFTED.labels(pipeline=p).set(0)
        BEST_VAL_ACCURACY.labels(pipeline=p).set(0)
        TRAINING_SLOTS_USED.labels(pipeline=p).set(0)
        MODEL_REGISTRY_VERSIONS.labels(pipeline=p, stage="staging").set(0)
        MODEL_REGISTRY_VERSIONS.labels(pipeline=p, stage="production").set(0)
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
