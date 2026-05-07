from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.prediction import PredictionStatus
from app.predictions.service import PredictionService


class DummyRepo:
    def __init__(self):
        self.created = None
        self.updated = None

    async def create(self, prediction):
        self.created = prediction
        prediction.id = getattr(prediction, "id", None) or SimpleNamespace()
        return prediction

    async def update(self, prediction):
        self.updated = prediction
        return prediction


class DummyPatientRepo:
    async def get_by_id(self, _id):
        return SimpleNamespace(
            first_name="A", age=70, gender=SimpleNamespace(value="M")
        )


class DummyScanRepo:
    async def get_by_id(self, _id):
        return SimpleNamespace(left_scan_path="left.png", right_scan_path="right.png")


@pytest.mark.asyncio
async def test_prediction_service_persists_embedding_into_output_payload(monkeypatch):
    service = PredictionService.__new__(PredictionService)
    service.db = AsyncMock()
    service.repo = DummyRepo()  # type: ignore
    service.patient_repo = DummyPatientRepo()  # type: ignore
    service.mri_scan_repo = DummyScanRepo()  # type: ignore

    prediction = SimpleNamespace(
        id="pred-1",
        input_payload={"risk_factors": []},
        patient_id="patient-1",
        output_payload=None,
        status=PredictionStatus.PENDING,
    )

    service.repo.create = AsyncMock(return_value=prediction)
    service.repo.update = AsyncMock(return_value=prediction)

    monkeypatch.setattr(
        "app.predictions.service.ml_client.predict",
        AsyncMock(
            return_value=SimpleNamespace(
                prediction={"combined_grade": 2},
                confidence_score=0.93,
                model_name="efficientnet_b4",
                model_version="v1",
                embedding=[0.1, 0.2, 0.3],
                gradcam_left=None,
                gradcam_right=None,
                regions_left=None,
                regions_right=None,
                top_hotspots_left=None,
                top_hotspots_right=None,
                shap_explanation=None,
            )
        ),
    )
    monkeypatch.setattr(
        "app.predictions.service.get_socket_manager",
        lambda: SimpleNamespace(emit_prediction_event=AsyncMock()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.explanations.service",
        SimpleNamespace(
            ExplanationService=lambda _db: SimpleNamespace(
                trigger_xai_for_prediction=AsyncMock()
            )
        ),
    )

    result = await PredictionService.run(
        service,
        SimpleNamespace(
            patient_id="patient-1",
            mri_scan_id="scan-1",
            model_name="efficientnet_b4",
            model_version="v1",
            input_payload={"risk_factors": []},
        ),  # type: ignore
        requested_by="user-1",  # type: ignore
    )

    assert result.output_payload is not None
    assert result.output_payload["embedding"] == [0.1, 0.2, 0.3]
