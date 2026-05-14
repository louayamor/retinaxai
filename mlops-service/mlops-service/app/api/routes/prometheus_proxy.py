from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_settings

router = APIRouter(prefix="/metrics", tags=["metrics"])


class AlertItemResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    labels: dict[str, str]
    annotations: dict[str, str]
    starts_at: str = Field(alias="startsAt")
    ends_at: str | None = Field(default=None, alias="endsAt")
    status: str
    value: str | None = None
    fingerprint: str | None = None


class AlertsResponse(BaseModel):
    alerts: list[AlertItemResponse]
    total: int
    firing: int
    pending: int


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
async def get_alerts():
    settings = get_settings()
    prometheus_url = settings.prometheus_url

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{prometheus_url}/api/v1/alerts")
            data: dict[str, Any] = resp.json()

            if data.get("status") != "success":
                raise HTTPException(
                    status_code=502,
                    detail="Prometheus alerts API returned non-success status",
                )

            raw_alerts: list[dict[str, Any]] = data.get("data", {}).get("alerts", [])

            mapped: list[AlertItemResponse] = []
            firing = 0
            pending = 0

            for raw in raw_alerts:
                state = raw.get("state", "inactive")
                value_str: str | None = raw.get("value")
                if value_str is not None:
                    value_str = str(value_str)

                alert_item = AlertItemResponse(
                    labels=raw.get("labels", {}),
                    annotations=raw.get("annotations", {}),
                    starts_at=raw.get("activeAt", ""),
                    ends_at=None,
                    status=state,
                    value=value_str,
                    fingerprint=raw.get("fingerprint"),
                )

                if state == "firing":
                    firing += 1
                elif state == "pending":
                    pending += 1

                mapped.append(alert_item)

            mapped.sort(key=lambda a: a.starts_at, reverse=True)

            return AlertsResponse(
                alerts=mapped,
                total=len(mapped),
                firing=firing,
                pending=pending,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prometheus unreachable: {e}")
