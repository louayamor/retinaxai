import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.prediction_explanation import ExplanationStatus, PredictionExplanation
from app.models.gradcam_explanation import GradCAMExplanation
from app.models.severity_report import RiskLevel, SeverityReport
from app.predictions.repository import PredictionRepository

router = APIRouter(prefix="/explanations", tags=["explanations"])


class StoreXAIRequest(BaseModel):
    prediction_id: str
    explanation_content: str | None = None
    explanation_summary: str | None = None
    explanation_model: str | None = None
    shap_values: dict[str, Any] | None = None
    gradcam_left_explanation: str | None = None
    gradcam_right_explanation: str | None = None
    severity_content: str | None = None
    severity_summary: str | None = None
    severity_risk_level: str = "moderate"
    severity_recommendations: list[str] = []


class StoreXAIShapRequest(BaseModel):
    prediction_id: str
    shap_values: dict[str, Any]
    content: str | None = None
    summary: str | None = None
    model: str = "shap"


@router.post("/store")
async def store_xai_results(
    request: StoreXAIRequest,
    db: AsyncSession = Depends(get_db),
):
    """Store XAI results from LLMOps service."""
    from app.websockets.manager import get_socket_manager

    prediction_repo = PredictionRepository(db)
    prediction = await prediction_repo.get_by_id(uuid.UUID(request.prediction_id))

    if not prediction:
        return {"status": "error", "message": "Prediction not found"}

    results = {"stored": []}

    if request.explanation_content or request.shap_values:
        exp = PredictionExplanation(
            id=uuid.uuid4(),
            prediction_id=prediction.id,
            content=request.explanation_content or "",
            summary=request.explanation_summary,
            model_used=request.explanation_model or "unknown",
            status=ExplanationStatus.COMPLETED,
            shap_values=request.shap_values,
        )
        db.add(exp)
        results["stored"].append("prediction_explanation")

        prediction.output_payload = prediction.output_payload or {}
        if request.explanation_content:
            prediction.output_payload["explanation"] = request.explanation_content
        if request.shap_values:
            prediction.output_payload["shap_values"] = request.shap_values

    if request.gradcam_left_explanation or request.gradcam_right_explanation:
        gradcam_exp = GradCAMExplanation(
            id=uuid.uuid4(),
            prediction_id=prediction.id,
            left_eye_explanation=request.gradcam_left_explanation or "",
            right_eye_explanation=request.gradcam_right_explanation or "",
            highlighted_regions={},
            model_used=request.explanation_model or "unknown",
        )
        db.add(gradcam_exp)
        results["stored"].append("gradcam_explanation")

    if request.severity_content:
        risk_level = RiskLevel(request.severity_risk_level.lower())
        severity = SeverityReport(
            id=uuid.uuid4(),
            prediction_id=prediction.id,
            patient_id=prediction.patient_id,
            content=request.severity_content,
            summary=request.severity_summary,
            risk_level=risk_level,
            recommendations=request.severity_recommendations,
            model_used=request.explanation_model or "unknown",
        )
        db.add(severity)
        results["stored"].append("severity_report")

    await db.commit()

    try:
        socket_manager = get_socket_manager()
        asyncio.create_task(
            socket_manager.emit_xai_event(
                event_type="xai.explanation_ready",
                prediction_id=str(prediction.id),
                patient_id=str(prediction.patient_id),
                status="completed",
                progress=100,
                message="XAI explanation stored",
            )
        )
    except Exception:
        pass

    return {"status": "ok", "prediction_id": request.prediction_id, "results": results}


@router.post("/store/shap")
async def store_shap_results(
    request: StoreXAIShapRequest,
    db: AsyncSession = Depends(get_db),
):
    """Store SHAP explanation results."""
    prediction_repo = PredictionRepository(db)
    prediction = await prediction_repo.get_by_id(uuid.UUID(request.prediction_id))

    if not prediction:
        return {"status": "error", "message": "Prediction not found"}

    exp = PredictionExplanation(
        id=uuid.uuid4(),
        prediction_id=prediction.id,
        content=request.content or "",
        summary=request.summary,
        model_used=request.model,
        status=ExplanationStatus.COMPLETED,
        shap_values=request.shap_values,
    )
    db.add(exp)

    prediction.output_payload = prediction.output_payload or {}
    prediction.output_payload["shap_values"] = request.shap_values

    await db.commit()

    return {"status": "ok", "prediction_id": request.prediction_id}
