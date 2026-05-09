from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger


REPORT_REQUESTS_TOTAL = Counter(
    "retinaxai_llm_report_requests_total",
    "Total report generation requests",
    ["status"],
)

ASYNC_JOBS_TOTAL = Counter(
    "retinaxai_llm_async_jobs_total",
    "Total async job operations",
    ["status"],
)

ACTIVE_JOBS = Gauge(
    "retinaxai_llm_active_jobs",
    "Number of currently running async jobs",
)

LLM_GENERATION_LATENCY = Histogram(
    "retinaxai_llm_generation_latency_seconds",
    "LLM report generation latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RAG_RETRIEVAL_LATENCY = Histogram(
    "retinaxai_llm_rag_retrieval_latency_seconds",
    "RAG retrieval latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RAG_DOCUMENTS_RETRIEVED = Gauge(
    "retinaxai_llm_rag_documents_retrieved",
    "Number of documents retrieved from RAG per request",
)

SHAP_EXPLAIN_REQUESTS_TOTAL = Counter(
    "retinaxai_llm_shap_requests_total",
    "Total SHAP explanation requests",
    ["status"],
)

XAI_GRADCAM_REQUESTS_TOTAL = Counter(
    "retinaxai_xai_gradcam_requests_total",
    "Total GradCAM explanation requests",
    ["status"],
)

XAI_PREDICTION_REQUESTS_TOTAL = Counter(
    "retinaxai_xai_prediction_requests_total",
    "Total XAI prediction explanation requests",
    ["status"],
)

XAI_GRADCAM_LATENCY = Histogram(
    "retinaxai_xai_gradcam_latency_seconds",
    "GradCAM explanation generation latency in seconds",
    buckets=[2, 5, 10, 20, 30, 60],
)

XAI_PREDICTION_LATENCY = Histogram(
    "retinaxai_xai_prediction_latency_seconds",
    "XAI prediction explanation generation latency in seconds",
    buckets=[2, 5, 10, 20, 30, 60],
)

XAI_GRADCAM_REGION_COUNT = Gauge(
    "retinaxai_xai_gradcam_region_count",
    "Number of regions in GradCAM per eye",
    ["eye"],
)

XAI_RAG_AVAILABLE = Gauge(
    "retinaxai_xai_rag_available",
    "Whether RAG was available for the last GradCAM request",
)

XAI_GRADCAM_STRUCTURE_OK = Counter(
    "retinaxai_xai_gradcam_structure_ok",
    "GradCAM response passes structure validation",
    ["result"],
)


def start_metrics_server(port: int = 9092) -> None:
    try:
        start_http_server(port)
        logger.info(f"prometheus metrics server started on port {port}")
    except OSError as e:
        logger.warning(f"Could not start metrics server on port {port}: {e}")
