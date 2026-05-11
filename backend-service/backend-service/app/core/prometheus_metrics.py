from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger


REQUEST_COUNT = Counter(
    "retinaxai_backend_requests_total",
    "Total HTTP requests to backend",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "retinaxai_backend_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

ERROR_COUNT = Counter(
    "retinaxai_backend_errors_total",
    "Total errors in backend",
    ["error_type"],
)

ACTIVE_WEBSOCKET_CONNECTIONS = Gauge(
    "retinaxai_backend_websocket_connections",
    "Number of active WebSocket connections",
)

PREDICTIONS_TOTAL = Counter(
    "retinaxai_backend_predictions_total",
    "Total predictions requested",
    ["status"],
)

REPORTS_GENERATED_TOTAL = Counter(
    "retinaxai_backend_reports_generated_total",
    "Total reports generated",
    ["status"],
)


def start_metrics_server(port: int = 9102) -> None:
    try:
        start_http_server(port)
        logger.info(f"prometheus metrics server started on port {port}")
    except OSError as e:
        logger.warning(f"Could not start metrics server on port {port}: {e}")
