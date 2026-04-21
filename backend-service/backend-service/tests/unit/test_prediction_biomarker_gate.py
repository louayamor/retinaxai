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
        return {"biomarkers": {"tortuosity": 0.4, "vessel_density": 0.2}}


class DummyExplanationService:
    def __init__(self, _db):
        self.called = False

    async def trigger_xai_for_prediction(self, prediction, patient_data):
        self.called = True
        assert patient_data["vascular_biomarkers"]["tortuosity"] == 0.4
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_prediction_service_stores_biomarkers_before_xai(monkeypatch):
    service = PredictionService(db=SimpleNamespace())
    service.repo = DummyRepo()
    service.patient_repo = DummyPatientRepo()
    service.mri_scan_repo = DummyScanRepo()

    monkeypatch.setattr("app.predictions.service.ml_client", DummyMLClient())
    monkeypatch.setattr("app.predictions.service.biomarker_client", DummyBiomarkerClient())
    monkeypatch.setattr("app.predictions.service.get_socket_manager", lambda: SimpleNamespace(emit_prediction_event=lambda **_kwargs: None))

    fake_explanations_service = types.ModuleType("app.explanations.service")
    fake_explanations_service.ExplanationService = DummyExplanationService
    monkeypatch.setitem(sys.modules, "app.explanations.service", fake_explanations_service)

    data = SimpleNamespace(
        patient_id="patient-1",
        mri_scan_id="scan-1",
        model_name="model",
        model_version="v1",
        input_payload={},
    )

    prediction = await service.run(data, "user-1")
    await asyncio.sleep(0)

    assert prediction.status == PredictionStatus.SUCCESS
    assert prediction.output_payload["vascular_biomarkers"]["tortuosity"] == 0.4
