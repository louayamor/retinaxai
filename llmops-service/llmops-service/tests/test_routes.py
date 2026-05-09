from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.inference_pipeline import get_inference_pipeline
from app.pipeline.xai_pipeline import get_xai_pipeline
from app.services.job_manager import get_job_manager, JobManager
from app.services.operation_state import (
    get_operation_state_manager,
    OperationStateManager,
)
from app.services.shap_service import get_shap_service, ShapService
from app.services.websocket_client import get_websocket_client, WebSocketClient
from app.vectorstore.chroma_store import ChromaStore

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "model" in body


def test_generate_endpoint_with_mock(monkeypatch):
    class MockPipeline:
        async def generate_report(self, payload: dict) -> dict:
            return {
                "content": "Mock clinical report.",
                "summary": "Mock summary.",
                "model_used": "mock-model",
            }

    app.dependency_overrides[get_inference_pipeline] = lambda: MockPipeline()

    payload = {
        "patient": {"id": "P001", "age": 55},
        "prediction": {"grade": 2},
        "report_type": "report",
        "language": "en",
        "tone": "clinical",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    parsed = json.loads(body["response"])
    assert parsed["content"] == "Mock clinical report."
    assert parsed["summary"] == "Mock summary."

    del app.dependency_overrides[get_inference_pipeline]


def test_generate_endpoint_returns_503_on_error(monkeypatch):
    class BrokenPipeline:
        async def generate_report(self, payload: dict) -> dict:
            raise RuntimeError("LLM unavailable")

    app.dependency_overrides[get_inference_pipeline] = lambda: BrokenPipeline()

    response = client.post("/api/generate", json={})
    assert response.status_code == 503

    del app.dependency_overrides[get_inference_pipeline]


def test_rag_endpoints_use_indexing_pipeline(monkeypatch):
    import app.api.routes as routes_module

    class MockIndexingPipeline:
        def run(self) -> dict:
            return {"schema_version": "1.0", "run_id": "run-1", "artifact_count": 4}

    monkeypatch.setattr(routes_module, "IndexingPipeline", MockIndexingPipeline)

    response = client.post("/api/rag/reindex")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["result"]["run_id"] == "run-1"

    monkeypatch.setattr(
        routes_module.ChromaStore,
        "read_state",
        lambda self: {
            "schema_version": "1.0",
            "run_id": "run-1",
            "artifact_count": 4,
        },
    )
    status_response = client.get("/api/rag/status")
    assert status_response.status_code == 200
    assert status_response.json()["artifact_count"] == 4
    assert status_response.json()["status"] == "ready"


def test_job_status_endpoint_uses_dependency(monkeypatch):
    job_manager = JobManager()
    app.dependency_overrides[get_job_manager] = lambda: job_manager

    import asyncio

    job_id = asyncio.run(job_manager.submit("test", {"key": "val"}))

    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id

    del app.dependency_overrides[get_job_manager]


def test_operation_status_endpoint_uses_dependency():
    op_manager = OperationStateManager()
    op_manager.set_operation("idle", "Ready")
    app.dependency_overrides[get_operation_state_manager] = lambda: op_manager

    response = client.get("/api/operation/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "idle"
    assert body["message"] == "Ready"

    del app.dependency_overrides[get_operation_state_manager]
