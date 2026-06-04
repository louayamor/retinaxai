from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth.role_guard import StaffUser
from app.db.session import get_db
from app.models.prediction import Prediction
from app.explanations.service import ExplanationService

router = APIRouter(prefix="/explanations", tags=["explanations"])


class StoreXAIRequest(BaseModel):
    prediction_id: str
    explanation_content: str | None = None
    explanation_summary: str | None = None
    explanation_model: str | None = None
    gradcam_left_explanation: str | None = None
    gradcam_right_explanation: str | None = None
    severity_content: str | None = None
    severity_summary: str | None = None
    severity_risk_level: str = "moderate"
    severity_recommendations: list[str] = Field(default_factory=list)


@router.post("/store")
async def store_xai_results(
    request: StoreXAIRequest,
    _: StaffUser,
    db: AsyncSession = Depends(get_db),
):
    """Store XAI results from LLMOps service."""
    from sqlalchemy.exc import IntegrityError

    service = ExplanationService(db)
    try:
        result = await service.store_xai_results(
            prediction_id=uuid.UUID(request.prediction_id),
            explanation_content=request.explanation_content,
            explanation_summary=request.explanation_summary,
            explanation_model=request.explanation_model,
            gradcam_left_explanation=request.gradcam_left_explanation,
            gradcam_right_explanation=request.gradcam_right_explanation,
            severity_content=request.severity_content,
            severity_summary=request.severity_summary,
            severity_risk_level=request.severity_risk_level,
            severity_recommendations=request.severity_recommendations,
        )
        if result.get("status") == "error":
            return result
        return result
    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="XAI explanation already exists for this prediction",
        )


class XAIResponse(BaseModel):
    prediction_id: str
    explanation: dict | None = None
    severity_report: dict | None = None
    gradcam_explanation: dict | None = None


@router.get("/{prediction_id}", response_model=XAIResponse)
async def get_xai_explanations(
    prediction_id: str,
    _: StaffUser,
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
            selectinload(Prediction.gradcam_explanation),
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

    if prediction.gradcam_explanation:
        response["gradcam_explanation"] = {
            "id": str(prediction.gradcam_explanation.id),
            "left_eye_explanation": prediction.gradcam_explanation.left_eye_explanation,
            "right_eye_explanation": prediction.gradcam_explanation.right_eye_explanation,
            "highlighted_regions": prediction.gradcam_explanation.highlighted_regions,
            "model_used": prediction.gradcam_explanation.model_used,
        }

    return response
