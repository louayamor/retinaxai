from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_settings

router = APIRouter(prefix="/metrics", tags=["metrics"])


class PrometheusMetricsResponse(BaseModel):
    training_runs_total: float
    active_training_jobs: float
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
