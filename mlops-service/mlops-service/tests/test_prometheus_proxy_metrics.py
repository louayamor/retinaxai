from __future__ import annotations

import asyncio

from app.api.routes.prometheus_proxy import get_prometheus_metrics


class DummySettings:
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url


def test_prometheus_proxy_handles_missing_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.prometheus_proxy.get_settings",
        lambda: DummySettings("http://prometheus.test"),
    )

    class DummyResponse:
        def json(self):
            return {"status": "success", "data": {"result": []}}

    async def _fake_get(*_args, **_kwargs):
        return DummyResponse()

    monkeypatch.setattr("httpx.AsyncClient.get", _fake_get)

    async def run():
        response = await get_prometheus_metrics()
        assert response.cpu_utilization is None

    asyncio.run(run())
