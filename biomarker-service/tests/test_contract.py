from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_service_metadata():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "biomarker-service"


def test_extract_endpoint_accepts_multipart_form(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        "app.service.BiomarkerService.extract_biomarkers",
        lambda _self, _image_bytes: {
            "tortuosity": 0.1,
            "avr": 0.2,
            "fractal_dimension": 1.5,
            "vessel_density": 0.3,
            "bifurcation_count": 4,
            "bifurcation_angles": [15.0],
            "cre": {"artery_cre": 1.1, "vein_cre": 1.4},
            "raw_feature_vector": [0.1, 0.2],
        },
    )

    response = client.post(
        "/biomarkers/extract",
        data={
            "prediction_id": "pred-1",
            "patient_id": "patient-1",
            "eye_side": "left",
            "model_version": "v1",
        },
        files={"image": ("scan.png", b"fake-bytes", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "pred-1"
    assert body["patient_id"] == "patient-1"
    assert body["status"] == "success"
    assert body["contract_version"] == "0.1.0"
    assert body["biomarkers"]["tortuosity"] == 0.1
