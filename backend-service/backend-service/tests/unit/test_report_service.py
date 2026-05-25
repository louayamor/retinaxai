from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.models.prediction import PredictionStatus
from app.reports.service import ReportService


class DummyPredictionRepo:
    async def get_by_id(self, _prediction_id):
        return SimpleNamespace(
            id="pred-1",
            patient_id="patient-1",
            model_name="model",
            model_version="v1",
            confidence_score=0.91,
            status=PredictionStatus.SUCCESS,
            output_payload={"combined_grade": 3, "overall_severity": "high"},
            requested_by="user-1",
        )


class DummyPatientRepo:
    async def get_by_id(self, _patient_id):
        return SimpleNamespace(
            id="patient-1",
            first_name="Louay",
            last_name="Test",
            age=42,
            gender=SimpleNamespace(value="M"),
            medical_record_number="mrn-1",
        )


class DummyReportRepo:
    async def get_by_prediction_id(self, _prediction_id):
        return None

    async def create(self, report):
        return report

    async def update(self, report):
        return report


class DummyLLMClient:
    async def generate_report(self, _request):
        return SimpleNamespace(content="ok", summary="sum", model_used="model")


@pytest.mark.asyncio
async def test_report_service_generate_completes_without_background_task(monkeypatch):
    service = ReportService(db=SimpleNamespace(), llm_client_override=DummyLLMClient())
    service.prediction_repo = DummyPredictionRepo()
    service.patient_repo = DummyPatientRepo()
    service.repo = DummyReportRepo()

    result = await service.generate(SimpleNamespace(prediction_id="pred-1"), "user-1")

    assert result.content == "ok"
    assert result.summary == "sum"
