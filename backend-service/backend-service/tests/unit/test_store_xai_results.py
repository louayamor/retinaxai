from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.explanations.service import ExplanationService


class DummyDBXAI:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False
        self.rolled_back = False

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def flush(self) -> None:
        pass


@pytest.mark.asyncio
async def test_store_xai_results_prediction_not_found() -> None:
    db = DummyDBXAI()
    service = ExplanationService(db)

    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=None)

    with patch("app.explanations.service.PredictionRepository", return_value=mock_repo):
        result = await service.store_xai_results(
            prediction_id=uuid.uuid4(),
            explanation_content="test content",
        )

    assert result["status"] == "error"
    assert result["message"] == "Prediction not found"


@pytest.mark.asyncio
async def test_store_xai_results_stores_all_artifacts() -> None:
    db = DummyDBXAI()
    service = ExplanationService(db)

    pred_id = uuid.uuid4()
    mock_prediction = AsyncMock()
    mock_prediction.id = pred_id
    mock_prediction.patient_id = uuid.uuid4()
    mock_prediction.output_payload = None

    mock_repo = AsyncMock()
    mock_repo.get_by_id = AsyncMock(return_value=mock_prediction)

    with (
        patch("app.explanations.service.PredictionRepository", return_value=mock_repo),
        patch("app.explanations.service.emit_xai_event"),
    ):
        result = await service.store_xai_results(
            prediction_id=pred_id,
            explanation_content="Explanation text",
            explanation_summary="Summary text",
            explanation_model="gpt-4o",
            gradcam_left_explanation="Left eye findings",
            gradcam_right_explanation="Right eye findings",
            severity_content="Severity assessment",
            severity_summary="Mild",
            severity_risk_level="low",
            severity_recommendations=["monitor"],
        )

    assert result["status"] == "ok"
    assert "prediction_explanation" in result["results"]["stored"]
    assert "gradcam_explanation" in result["results"]["stored"]
    assert "severity_report" in result["results"]["stored"]
    assert db.committed
