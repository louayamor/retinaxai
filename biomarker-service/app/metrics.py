from __future__ import annotations

from prometheus_client import Counter, Histogram


EXTRACTION_REQUESTS_TOTAL = Counter(
    "retinaxai_biomarker_extraction_requests_total",
    "Total biomarker extraction requests",
    ["status"],
)

EXTRACTION_FAILURES_TOTAL = Counter(
    "retinaxai_biomarker_extraction_failures_total",
    "Total biomarker extraction failures",
    ["reason"],
)

EXTRACTION_DURATION_SECONDS = Histogram(
    "retinaxai_biomarker_extraction_duration_seconds",
    "Biomarker extraction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
