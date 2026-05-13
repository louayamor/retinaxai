from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_settings

router = APIRouter(prefix="/metrics", tags=["metrics"])


class PrometheusMetricsResponse(BaseModel):
    training_runs_total: float | None = None
    active_training_jobs: float | None = None
    best_val_accuracy_imaging: float | None
    best_val_accuracy_clinical: float | None
    drift_detected_imaging: float | None
    drift_detected_clinical: float | None
    evidently_dataset_shift_imaging: float | None
    evidently_dataset_shift_clinical: float | None
    evidently_features_drifted_imaging: float | None
    evidently_features_drifted_clinical: float | None
    inference_latency_p95: float | None
    gpu_utilization: float | None


class Alert(BaseModel):
    labels: dict[str, str]
    annotations: dict[str, str]
    startsAt: str
    endsAt: str | None = None
    generatorURL: str | None = None
    fingerprint: str | None = None
    status: str | None = None
    value: str | None = None


class AlertsResponse(BaseModel):
    alerts: list[Alert]
    total: int
    firing: int
    pending: int


@router.get("/prometheus", response_model=PrometheusMetricsResponse)
async def get_prometheus_metrics():
    """Proxy Prometheus metrics for frontend consumption."""
    settings = get_settings()
    prometheus_url = settings.prometheus_url

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            queries = {
                "training_runs_total": "sum(retinaxai_training_runs_total)",
                "active_training_jobs": "retinaxai_active_training_jobs",
                "best_val_accuracy_imaging": "retinaxai_best_val_accuracy{pipeline='imaging'}",
                "best_val_accuracy_clinical": "retinaxai_best_val_accuracy{pipeline='clinical'}",
                "drift_detected_imaging": "retinaxai_drift_detected{pipeline='imaging'}",
                "drift_detected_clinical": "retinaxai_drift_detected{pipeline='clinical'}",
                "evidently_dataset_shift_imaging": "retinaxai_evidently_dataset_shift{pipeline='imaging'}",
                "evidently_dataset_shift_clinical": "retinaxai_evidently_dataset_shift{pipeline='clinical'}",
                "evidently_features_drifted_imaging": "retinaxai_evidently_features_drifted{pipeline='imaging'}",
                "evidently_features_drifted_clinical": "retinaxai_evidently_features_drifted{pipeline='clinical'}",
                "inference_latency_p95": "histogram_quantile(0.95, sum(rate(retinaxai_inference_latency_seconds_bucket[5m])) by (le))",
                "gpu_utilization": "retinaxai_gpu_utilization_percent",
            }

            results = {}
            for name, query in queries.items():
                try:
                    resp = await client.get(
                        f"{prometheus_url}/api/v1/query",
                        params={"query": query},
                    )
                    data = resp.json()
                    if data.get("status") == "success" and data.get("data", {}).get(
                        "result"
                    ):
                        val = float(data["data"]["result"][0]["value"][1])
                        results[name] = val
                    else:
                        results[name] = None
                except Exception:
                    results[name] = None

            return PrometheusMetricsResponse(**results)

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prometheus unreachable: {e}")


@router.get("/alerts", response_model=AlertsResponse)
async def get_prometheus_alerts():
    """Fetch active alerts from Prometheus rule evaluation."""
    settings = get_settings()
    prometheus_url = settings.prometheus_url

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{prometheus_url}/api/v1/alerts")
            data = resp.json()

            if data.get("status") != "success":
                return AlertsResponse(alerts=[], total=0, firing=0, pending=0)

            raw_alerts = data.get("data", {}).get("alerts", [])
            alerts: list[Alert] = []
            firing_count = 0
            pending_count = 0

            for raw in raw_alerts:
                labels = raw.get("labels", {})
                annotations = raw.get("annotations", {})
                state = raw.get("state", "unknown")

                if state == "firing":
                    firing_count += 1
                elif state == "pending":
                    pending_count += 1

                alerts.append(
                    Alert(
                        labels=labels,
                        annotations=annotations,
                        startsAt=raw.get("startsAt", ""),
                        endsAt=raw.get("endsAt"),
                        generatorURL=raw.get("generatorURL"),
                        fingerprint=raw.get("fingerprint"),
                        status=state,
                        value=raw.get("value"),
                    )
                )

            return AlertsResponse(
                alerts=alerts,
                total=len(alerts),
                firing=firing_count,
                pending=pending_count,
            )

    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Prometheus alerts unreachable: {e}"
        )
