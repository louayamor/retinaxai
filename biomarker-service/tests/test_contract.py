from __future__ import annotations

import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.service import BiomarkerExtractionError, BiomarkerService


def test_health_endpoint_returns_service_metadata():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "biomarker-service"


def test_extract_endpoint_accepts_multipart_form(monkeypatch):
    class DummyAdapter:
        def predict(self, _image_bytes):
            return {
                "tortuosity": 0.1,
                "avr": 0.2,
                "fractal_dimension": 1.5,
                "vessel_density": 0.3,
                "bifurcation_count": 4,
                "bifurcation_angles": [15.0],
                "cre": {"artery_cre": 1.1, "vein_cre": 1.4},
                "raw_feature_vector": [0.1, 0.2],
            }

    monkeypatch.setattr("app.service.VascXRegistry.load", lambda _self: DummyAdapter())

    client = TestClient(app)

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
    assert body["contract_version"] == "1.0"
    assert body["biomarkers"]["tortuosity"] == 0.1


def test_service_extracts_biomarkers_from_valid_image():
    service = BiomarkerService()

    class DummyAdapter:
        def predict(self, _image_bytes):
            return {
                "tortuosity": 0.1,
                "avr": 0.2,
                "fractal_dimension": 1.5,
                "vessel_density": 0.3,
                "bifurcation_count": 4,
                "bifurcation_angles": [15.0],
                "cre": {"artery_cre": 1.1, "vein_cre": 1.4},
                "raw_feature_vector": [0.1, 0.2],
            }

    service._registry.load = lambda: DummyAdapter()
    image = Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    biomarkers = service.extract_biomarkers(buffer.getvalue())

    assert biomarkers.bifurcation_angles is not None
    assert biomarkers.raw_feature_vector
    assert biomarkers.vessel_density is not None


def test_service_rejects_empty_payload():
    service = BiomarkerService()

    class DummyAdapter:
        def predict(self, _image_bytes):
            raise BiomarkerExtractionError("empty image payload")

    service._registry.load = lambda: DummyAdapter()

    try:
        service.extract_biomarkers(b"")
    except BiomarkerExtractionError as exc:
        assert "empty image payload" in str(exc)
    else:
        raise AssertionError("Expected BiomarkerExtractionError")
