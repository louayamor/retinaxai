from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.dependencies import get_settings


class DummySettings:
    def __init__(self, base: Path):
        self.ocr_output_dir = base / "ocr"
        self.clinical_metrics_path = base / "clinical_metrics.json"
        self.clinical_feature_importance_path = base / "clinical_feature_importance.json"
        self.imaging_metrics_path = base / "imaging_metrics.json"


def test_rag_manifest_endpoint_returns_indexable_artifacts(tmp_path, monkeypatch):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "reports.json").write_text("[{\"source_file\": \"scan.jpg\"}]")
    (tmp_path / "clinical_metrics.json").write_text('{"accuracy": 0.9}')
    (tmp_path / "clinical_feature_importance.json").write_text('{"age": 0.4}')
    (tmp_path / "imaging_metrics.json").write_text('{"eyepacs_test": {"accuracy": 0.8}}')

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: DummySettings(tmp_path)
    client = TestClient(app)

    response = client.get("/rag/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body["artifact_count"] == 4
    assert body["pipeline"] == "combined"
    assert body["schema_version"] == "1.0"
    assert {artifact["artifact_id"] for artifact in body["artifacts"]} == {
        "ocr_reports",
        "clinical_metrics",
        "clinical_feature_importance",
        "imaging_metrics",
    }
    assert all(artifact["schema_version"] == "1.0" for artifact in body["artifacts"])
    assert all(artifact["artifact_type"] == "json" for artifact in body["artifacts"])


def test_rag_manifest_endpoint_rejects_partial_bundle(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "reports.json").write_text("[{\"source_file\": \"scan.jpg\"}]")

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: DummySettings(tmp_path)
    client = TestClient(app)

    response = client.get("/rag/manifest")

    assert response.status_code == 200
    body = response.json()
    assert body["artifact_count"] == 1
    assert body["pipeline"] == "partial"
    assert {artifact["artifact_id"] for artifact in body["artifacts"]} == {"ocr_reports"}
