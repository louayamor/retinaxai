from __future__ import annotations
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.prometheus_metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/metrics", "/health"):
            return await call_next(request)

        method = request.method
        path = self._normalize_path(request.url.path)
        start = time.perf_counter()
        status_code = "500"
        error_type = None

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            error_type = "unhandled"
            ERROR_COUNT.labels(error_type=error_type).inc()
            raise
        else:
            if response.status_code >= 500:
                error_type = "server_error"
            elif response.status_code >= 400:
                error_type = "client_error"
            if error_type:
                ERROR_COUNT.labels(error_type=error_type).inc()
        finally:
            duration = time.perf_counter() - start
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

        return response

    def _normalize_path(self, path: str) -> str:
        parts = path.strip("/").split("/")
        normalized = []
        for part in parts:
            if self._is_uuid(part) or self._is_int(part):
                normalized.append(":id")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized)

    def _is_uuid(self, s: str) -> bool:
        return len(s) == 36 and s.count("-") == 4

    def _is_int(self, s: str) -> bool:
        return s.lstrip("-").isdigit()
