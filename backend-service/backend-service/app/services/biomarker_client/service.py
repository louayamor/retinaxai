from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import settings
from app.core.prometheus_metrics import (
    BIOMARKER_CIRCUIT_BREAKER_STATE,
    BIOMARKER_EXTRACTION_DURATION_SECONDS,
    BIOMARKER_EXTRACTION_FAILURES_TOTAL,
    BIOMARKER_EXTRACTION_REQUESTS_TOTAL,
    BIOMARKER_RETRY_ATTEMPTS_TOTAL,
)
from app.services.biomarker_client.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


class BiomarkerExtractionError(RuntimeError):
    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


@dataclass(slots=True)
class BiomarkerResilienceConfig:
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 8.0
    jitter: float = 0.2
    circuit_breaker_failures: int = 5
    circuit_breaker_recovery_seconds: float = 60.0
    circuit_breaker_half_open_successes: int = 2


class BiomarkerServiceClient:
    def __init__(self) -> None:
        self.base_url = settings.BIOMARKER_SERVICE_URL
        self.timeout = settings.BIOMARKER_SERVICE_TIMEOUT
        self.headers = {}
        if settings.BIOMARKER_SERVICE_API_KEY:
            self.headers["Authorization"] = f"Bearer {settings.BIOMARKER_SERVICE_API_KEY}"
        self._resilience = BiomarkerResilienceConfig(timeout_seconds=float(self.timeout))
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._resilience.circuit_breaker_failures,
            recovery_timeout=self._resilience.circuit_breaker_recovery_seconds,
            half_open_successes=self._resilience.circuit_breaker_half_open_successes,
        )

    @property
    def circuit_state(self) -> str:
        return self._circuit_breaker.current_state()

    async def extract_from_scan_path(
        self,
        *,
        scan_path: str,
        prediction_id: str,
        patient_id: str,
        eye_side: str,
        model_version: str,
    ) -> dict:
        path = Path(scan_path)
        if not path.exists():
            raise BiomarkerExtractionError(
                f"scan path not found: {scan_path}",
                error_code="BIOMARKER_SCAN_NOT_FOUND",
            )

        if not self._circuit_breaker.can_attempt():
            BIOMARKER_CIRCUIT_BREAKER_STATE.set(1)
            raise BiomarkerExtractionError(
                "biomarker circuit breaker open",
                error_code="BIOMARKER_CIRCUIT_OPEN",
            )

        BIOMARKER_CIRCUIT_BREAKER_STATE.set(0)

        payload = {
            "prediction_id": prediction_id,
            "patient_id": patient_id,
            "eye_side": eye_side,
            "model_version": model_version,
        }
        image_bytes = await asyncio.to_thread(path.read_bytes)
        files = {
            "image": (path.name, image_bytes, "application/octet-stream"),
        }

        async def _call():
            try:
                async with httpx.AsyncClient(timeout=self._resilience.timeout_seconds) as client:
                    response = await client.post(
                        f"{self.base_url}/biomarkers/extract",
                        data=payload,
                        files=files,
                        headers=self.headers,
                    )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "[BIOMARKER CLIENT] HTTP %s: %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                raise BiomarkerExtractionError(
                    f"biomarker service returned {exc.response.status_code}",
                    error_code="BIOMARKER_HTTP_ERROR",
                ) from exc
            except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
                logger.error("[BIOMARKER CLIENT] timeout: %s", exc)
                raise BiomarkerExtractionError("biomarker request timed out", error_code="BIOMARKER_TIMEOUT") from exc
            except Exception as exc:
                logger.error("[BIOMARKER CLIENT] extraction failed: %s", exc)
                raise BiomarkerExtractionError(str(exc), error_code="BIOMARKER_ERROR") from exc

        start = asyncio.get_event_loop().time()

        def _on_retry(exc: Exception, attempt: int, delay: float) -> None:
            BIOMARKER_RETRY_ATTEMPTS_TOTAL.inc()
            logger.warning(
                "[BIOMARKER CLIENT] retrying attempt=%s delay=%.2f error=%s",
                attempt + 1,
                delay,
                exc,
            )

        try:
            BIOMARKER_EXTRACTION_REQUESTS_TOTAL.labels(status="started").inc()
            response = await retry_with_backoff(
                _call,
                attempts=self._resilience.max_attempts,
                base_delay=self._resilience.base_delay,
                multiplier=self._resilience.multiplier,
                max_delay=self._resilience.max_delay,
                jitter=self._resilience.jitter,
                on_retry=_on_retry,
            )
            self._circuit_breaker.on_success()
            BIOMARKER_EXTRACTION_REQUESTS_TOTAL.labels(status="success").inc()
            return response
        except Exception as exc:
            self._circuit_breaker.on_failure()
            BIOMARKER_EXTRACTION_FAILURES_TOTAL.labels(error_type=type(exc).__name__).inc()
            raise
        finally:
            duration = asyncio.get_event_loop().time() - start
            BIOMARKER_EXTRACTION_DURATION_SECONDS.observe(duration)


biomarker_client = BiomarkerServiceClient()
