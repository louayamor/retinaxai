from __future__ import annotations
import re
import time as time_mod
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from app.auth.role_guard import EngineerUser
from app.db.session import engine
from app.services.redis_client import redis_client

router = APIRouter(prefix="/system", tags=["system"])

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:4000"
GRAFANA_USER = "admin"
GRAFANA_PASS = "prtgrm1998"

_RELATIVE_RE = re.compile(r"^(\d+)([mhdw])$")


def _resolve_time(t: str) -> float:
    """Convert relative time strings (6h, 30m, 7d, 1w) to absolute Unix timestamps."""
    if t == "now":
        return time_mod.time()
    m = _RELATIVE_RE.match(t)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        return time_mod.time() - value * multipliers[unit]
    try:
        return float(t)
    except ValueError:
        return time_mod.time()


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


async def _prom_query_range(
    query: str, start: str = "6h", end: str = "now", step: str = "5m"
) -> list[dict[str, Any]]:
    start_ts = _resolve_time(start)
    end_ts = _resolve_time(end)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": query, "start": start_ts, "end": end_ts, "step": step},
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


@router.get("/prometheus/range")
async def get_prometheus_range(
    query: str,
    start: str = "6h",
    end: str = "now",
    step: str = "5m",
) -> list[dict[str, Any]]:
    try:
        return await _prom_query_range(query, start=start, end=end, step=step)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.api_route("/grafana/proxy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def grafana_proxy(
    path: str,
    request: Request,
    _: EngineerUser,
):
    target_path = path or ""
    query_string = str(request.query_params)
    target_url = f"{GRAFANA_URL}/{target_path}"
    if query_string:
        target_url += f"?{query_string}"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None

        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "connection", "content-length", "transfer-encoding")
        }
        headers.pop("x-frame-options", None)

        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            auth=httpx.BasicAuth(GRAFANA_USER, GRAFANA_PASS),
        )

        content_type = resp.headers.get("content-type", "")
        content = resp.content

        if "text/html" in content_type:
            base_tag = f'<base href="/api/v1/system/grafana/proxy/">'
            content = content.replace(b"<head>", f"<head>{base_tag}".encode())

        response_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("x-frame-options", "content-security-policy", "content-encoding", "transfer-encoding", "content-length")
        }

        return Response(
            content=content,
            status_code=resp.status_code,
            headers=response_headers,
        )
