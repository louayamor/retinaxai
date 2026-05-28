from __future__ import annotations
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from app.auth.role_guard import EngineerUser
from app.db.session import engine
from app.services.redis_client import redis_client

router = APIRouter(prefix="/system", tags=["system"])

PROMETHEUS_URL = "http://localhost:9090"


async def _prom_query(query: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise HTTPException(status_code=502, detail=f"Prometheus error: {data}")
        return data["data"]["result"]


async def _prom_query_range(query: str, step: str = "1h") -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": "6h", "end": "now", "step": step},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise HTTPException(status_code=502, detail=f"Prometheus error: {data}")
        return data["data"]["result"]


async def _ping_service(url: str, timeout: float = 3.0) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/health")
            return "healthy" if resp.status_code < 500 else "unhealthy"
    except Exception:
        return None


@router.get("/health")
async def get_system_health(_: EngineerUser):
    redis_ok: str | None = None
    try:
        conn = await redis_client.get_connection()
        if conn:
            await conn.ping()
            redis_ok = "healthy"
    except Exception:
        redis_ok = None

    pg_ok: str | None = None
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            pg_ok = "healthy"
    except Exception:
        pg_ok = None

    mlops_status = await _ping_service("http://localhost:8004")
    llmops_status = await _ping_service("http://localhost:8002")

    return {
        "backend": "healthy",
        "mlops": mlops_status,
        "llmops": llmops_status,
        "redis": redis_ok,
        "postgres": pg_ok,
    }


@router.get("/metrics")
async def get_system_metrics(
    _: EngineerUser,
):
    cpu_usage_pct = 0.0
    try:
        results = await _prom_query(
            '100 * (1 - avg by (instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])))'
        )
        if results:
            cpu_usage_pct = round(float(results[0]["value"][1]), 1)
    except Exception:
        pass

    mem_total_gb = 0.0
    mem_available_gb = 0.0
    try:
        results = await _prom_query("node_memory_MemTotal_bytes/1024/1024/1024")
        if results:
            mem_total_gb = round(float(results[0]["value"][1]), 1)
    except Exception:
        pass
    try:
        results = await _prom_query("node_memory_MemAvailable_bytes/1024/1024/1024")
        if results:
            mem_available_gb = round(float(results[0]["value"][1]), 1)
    except Exception:
        pass
    mem_used_gb = round(mem_total_gb - mem_available_gb, 1)
    mem_pct = round((mem_used_gb / mem_total_gb * 100), 1) if mem_total_gb > 0 else 0.0

    disk_total_gb = 0.0
    disk_free_gb = 0.0
    try:
        results = await _prom_query(
            'node_filesystem_size_bytes{mountpoint="/"} / 1024 / 1024 / 1024'
        )
        if results:
            disk_total_gb = round(float(results[0]["value"][1]), 1)
    except Exception:
        pass
    try:
        results = await _prom_query(
            'node_filesystem_avail_bytes{mountpoint="/"} / 1024 / 1024 / 1024'
        )
        if results:
            disk_free_gb = round(float(results[0]["value"][1]), 1)
    except Exception:
        pass
    disk_used_gb = round(disk_total_gb - disk_free_gb, 1)
    disk_pct = round((disk_used_gb / disk_total_gb * 100), 1) if disk_total_gb > 0 else 0.0

    load_avg = 0.0
    try:
        results = await _prom_query("node_load1")
        if results:
            load_avg = round(float(results[0]["value"][1]), 2)
    except Exception:
        pass

    net_rx_mb = 0.0
    net_tx_mb = 0.0
    try:
        results = await _prom_query(
            'rate(node_network_receive_bytes_total{device!="lo"}[5m]) * 8 / 1024 / 1024'
        )
        for r in results:
            net_rx_mb += float(r["value"][1])
    except Exception:
        pass
    try:
        results = await _prom_query(
            'rate(node_network_transmit_bytes_total{device!="lo"}[5m]) * 8 / 1024 / 1024'
        )
        for r in results:
            net_tx_mb += float(r["value"][1])
    except Exception:
        pass

    return {
        "cpu": {"usage_percent": cpu_usage_pct},
        "memory": {
            "total_gb": mem_total_gb,
            "used_gb": mem_used_gb,
            "available_gb": mem_available_gb,
            "usage_percent": mem_pct,
        },
        "disk": {
            "total_gb": disk_total_gb,
            "used_gb": disk_used_gb,
            "free_gb": disk_free_gb,
            "usage_percent": disk_pct,
        },
        "load": load_avg,
        "network": {
            "rx_mbps": round(net_rx_mb, 2),
            "tx_mbps": round(net_tx_mb, 2),
        },
    }


@router.get("/gpu")
async def get_gpu_metrics(
    _: EngineerUser,
):
    gpu_info = []
    try:
        fields = [
            ("utilization", "nvidia_gpu_utilization", "value"),
            ("memory_used_gb", "nvidia_gpu_memory_used_bytes/1024/1024/1024", "value"),
            ("memory_total_gb", "nvidia_gpu_memory_total_bytes/1024/1024/1024", "value"),
            ("temperature_c", "nvidia_gpu_temperature_celsius", "value"),
            ("power_w", "nvidia_gpu_power_watts", "value"),
            ("fan_speed_pct", "nvidia_gpu_fan_speed_percent", "value"),
            ("clock_sm_mhz", "nvidia_gpu_clock_sm_mhz", "value"),
            ("clock_mem_mhz", "nvidia_gpu_clock_memory_mhz", "value"),
        ]

        results_map: dict[str, list] = {}
        for name, query, _ in fields:
            try:
                results = await _prom_query(query)
                for r in results:
                    gpu = r["metric"].get("gpu_name", "unknown")
                    results_map.setdefault(gpu, {})[name] = float(r["value"][1])
                    results_map[gpu]["name"] = gpu.replace("_", " ")
            except Exception:
                pass

        for gpu_name, metrics in results_map.items():
            mem_used = metrics.get("memory_used_gb", 0)
            mem_total = metrics.get("memory_total_gb", 0)
            gpu_info.append({
                "name": metrics.get("name", gpu_name),
                "utilization_pct": round(metrics.get("utilization", 0), 1),
                "memory_used_gb": round(mem_used, 2),
                "memory_total_gb": round(mem_total, 2),
                "memory_pct": round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0,
                "temperature_c": round(metrics.get("temperature_c", 0), 0),
                "power_w": round(metrics.get("power_w", 0), 1),
                "fan_speed_pct": round(metrics.get("fan_speed_pct", 0), 0),
                "clock_sm_mhz": round(metrics.get("clock_sm_mhz", 0), 0),
                "clock_mem_mhz": round(metrics.get("clock_mem_mhz", 0), 0),
            })
    except Exception:
        pass

    return {"gpus": gpu_info}
