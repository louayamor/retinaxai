"""
Template file for the RetinaXAI Biomarker Service.

This file contains project metadata and configuration constants.
"""

PROJECT_NAME = "RetinaXAI Biomarker Service"
PROJECT_VERSION = "0.1.0"

BIOMARKER_CONTRACT_VERSION = "1.0"

SERVICE_NAME = "biomarker-service"
SERVICE_PORT = 8010

EXTRACTION_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 8
JITTER_FACTOR = 0.2

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS = 60
CIRCUIT_BREAKER_HALF_OPEN_SUCCESSES = 2

HF_HOME = "/app/.cache/huggingface"
TRANSFORMERS_CACHE = "/app/.cache/huggingface"
RTNLS_MODEL_RELEASES = "/app/.cache/retinalysis-models"

LOG_LEVEL = "INFO"

HEALTH_ENDPOINT = "/health"
READY_ENDPOINT = "/ready"
METRICS_ENDPOINT = "/metrics"
