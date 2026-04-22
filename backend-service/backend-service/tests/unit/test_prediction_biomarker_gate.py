from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from app.models.prediction import PredictionStatus
from app.predictions.service import PredictionService


class DummyRepo:
    def __init__(self):
        self.created = None

    async def create(self, prediction):
        self.created = prediction
        return prediction

    async def update(self, prediction):
        return prediction


class DummyPatientRepo:
    async def get_by_id(self, _patient_id):
        return SimpleNamespace(first_name="Louay", last_name="Test", age=42, gender=SimpleNamespace(value="M"))


class DummyScanRepo:
    async def get_by_id(self, _scan_id):
        return SimpleNamespace(left_scan_path="/tmp/left.png", right_scan_path="/tmp/right.png")


class DummySocketManager:
    async def emit_prediction_event(self, **_kwargs):
        return None


class DummyMLResponse:
    prediction = {"combined_grade": 3}
    confidence_score = 0.88
    embedding = None
    gradcam_left = "left"
    gradcam_right = "right"
    regions_left = None
    regions_right = None
    top_hotspots_left = None
    top_hotspots_right = None
    shap_explanation = None


class DummyMLClient:
    async def predict(self, _request):
        return DummyMLResponse()


class DummyBiomarkerClient:
    async def extract_from_scan_path(self, **_kwargs):
        eye_side = _kwargs.get("eye_side")
        if eye_side == "left":
            return {"biomarkers": {"tortuosity": 0.4, "vessel_density": 0.2}}
        return {"biomarkers": {"tortuosity": 0.6, "vessel_density": 0.3}}


class FailingBiomarkerClient:
    async def extract_from_scan_path(self, **_kwargs):
        raise RuntimeError("biomarker extraction failed")


class DummyExplanationService:
    def __init__(self, _db):
        self.called = False

    async def trigger_xai_for_prediction(self, prediction, patient_data):
        self.called = True
        assert patient_data["vascular_biomarkers"]["left_eye"]["tortuosity"] == 0.4
        assert patient_data["vascular_biomarkers"]["right_eye"]["tortuosity"] == 0.6
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_prediction_service_stores_biomarkers_before_xai(monkeypatch):
    service = PredictionService(db=SimpleNamespace())
    service.repo = DummyRepo()
    service.patient_repo = DummyPatientRepo()
    service.mri_scan_repo = DummyScanRepo()

    monkeypatch.setattr("app.predictions.service.ml_client", DummyMLClient())
    monkeypatch.setattr("app.predictions.service.biomarker_client", DummyBiomarkerClient())
    monkeypatch.setattr("app.predictions.service.get_socket_manager", lambda: DummySocketManager())

    captured_explanation_service: dict[str, DummyExplanationService] = {}

    def explanation_service_factory(db):
        instance = DummyExplanationService(db)
        captured_explanation_service["instance"] = instance
        return instance

    fake_explanations_service = types.ModuleType("app.explanations.service")
    fake_explanations_service.ExplanationService = explanation_service_factory
    monkeypatch.setitem(sys.modules, "app.explanations.service", fake_explanations_service)

    data = SimpleNamespace(
        patient_id="patient-1",
        mri_scan_id="scan-1",
        model_name="model",
        model_version="v1",
        input_payload={},
    )

    prediction = await service.run(data, "user-1")

    assert prediction.status == PredictionStatus.SUCCESS
    assert prediction.output_payload["vascular_biomarkers_left"]["tortuosity"] == 0.4
    assert prediction.output_payload["vascular_biomarkers_right"]["tortuosity"] == 0.6
    assert captured_explanation_service["instance"].called is True


@pytest.mark.asyncio
async def test_prediction_service_keeps_success_when_biomarkers_fail(monkeypatch):
    service = PredictionService(db=SimpleNamespace())
    service.repo = DummyRepo()
    service.patient_repo = DummyPatientRepo()
    service.mri_scan_repo = DummyScanRepo()

    monkeypatch.setattr("app.predictions.service.ml_client", DummyMLClient())
    monkeypatch.setattr("app.predictions.service.biomarker_client", FailingBiomarkerClient())
    monkeypatch.setattr("app.predictions.service.get_socket_manager", lambda: DummySocketManager())

    data = SimpleNamespace(
        patient_id="patient-1",
        mri_scan_id="scan-1",
        model_name="model",
        model_version="v1",
        input_payload={},
    )

    prediction = await service.run(data, "user-1")

    assert prediction.status == PredictionStatus.SUCCESS
    assert prediction.biomarker_status.value == "FAILED"
    assert "biomarker extraction failed" in (prediction.biomarker_error_message or "")
