import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.prediction_explanation import ExplanationStatus, PredictionExplanation
from app.models.gradcam_explanation import GradCAMExplanation
from app.models.severity_report import RiskLevel, SeverityReport
from app.models.prediction import Prediction
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
            status="completed",
            shap_values=request.shap_values,
        )
        try:
            db.add(exp)
            await db.flush()
            results["stored"].append("prediction_explanation")
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=409,
                detail="XAI explanation already exists for this prediction",
            )

        prediction.output_payload = prediction.output_payload or {}
        if request.explanation_content:
            prediction.output_payload["explanation"] = request.explanation_content
        if request.shap_values:
            prediction.output_payload["shap_values"] = request.shap_values

        db.add(prediction)

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
        risk_level_str = (
            request.severity_risk_level.lower()
            if request.severity_risk_level
            else "moderate"
        )
        try:
            risk_enum = RiskLevel(risk_level_str)
        except ValueError:
            risk_enum = RiskLevel.MODERATE

        severity = SeverityReport(
            id=uuid.uuid4(),
            prediction_id=prediction.id,
            patient_id=prediction.patient_id,
            content=request.severity_content,
            summary=request.severity_summary,
            risk_level=risk_enum,
            recommendations=request.severity_recommendations or [],
            model_used=request.explanation_model or "unknown",
        )
        db.add(severity)
        results["stored"].append("severity_report")

        prediction.output_payload = prediction.output_payload or {}
        prediction.output_payload["severity_summary"] = request.severity_summary
        prediction.output_payload["severity_risk_level"] = risk_enum.value
        prediction.output_payload["severity_recommendations"] = (
            request.severity_recommendations or []
        )
        db.add(prediction)

    await db.commit()

    try:
        socket_manager = get_socket_manager()
        asyncio.create_task(
            socket_manager.emit_xai_event(
                event_type="xai.explanation_ready",
                prediction_id=str(prediction.id),
                patient_id=str(prediction.patient_id),
                status=ExplanationStatus.COMPLETED,
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


class XAIResponse(BaseModel):
    prediction_id: str
    explanation: dict | None = None
    severity_report: dict | None = None
    gradcam_explanation: dict | None = None


@router.get("/{prediction_id}")
async def get_xai_explanations(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve XAI explanations for a prediction."""
    try:
        pred_uuid = uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    prediction_stmt = (
        select(Prediction)
        .where(Prediction.id == pred_uuid)
        .options(
            selectinload(Prediction.explanation),
            selectinload(Prediction.severity_report),
        )
    )
    result = await db.execute(prediction_stmt)
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    response = {
        "prediction_id": prediction_id,
        "explanation": None,
        "severity_report": None,
        "gradcam_explanation": None,
    }

    if prediction.explanation:
        response["explanation"] = {
            "id": str(prediction.explanation.id),
            "content": prediction.explanation.content,
            "summary": prediction.explanation.summary,
            "model_used": prediction.explanation.model_used,
            "status": prediction.explanation.status.value,
            "shap_values": prediction.explanation.shap_values,
        }

    if prediction.severity_report:
        response["severity_report"] = {
            "id": str(prediction.severity_report.id),
            "content": prediction.severity_report.content,
            "summary": prediction.severity_report.summary,
            "risk_level": prediction.severity_report.risk_level.value,
            "recommendations": prediction.severity_report.recommendations,
            "model_used": prediction.severity_report.model_used,
        }

    return response
