from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint_exposes_biomarker_metrics():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "retinaxai_biomarker_extraction_requests_total" in body
    assert "retinaxai_biomarker_extraction_failures_total" in body
    assert "retinaxai_biomarker_extraction_duration_seconds" in body


def test_failed_extraction_updates_failure_metrics():
    client = TestClient(app)

    response = client.post(
        "/biomarkers/extract",
        data={"prediction_id": "pred-1", "patient_id": "patient-1"},
        files={"image": ("scan.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 500

    metrics = client.get("/metrics").text
    assert "reason=\"BiomarkerExtractionError\"" in metrics or "BiomarkerExtractionError" in metrics
