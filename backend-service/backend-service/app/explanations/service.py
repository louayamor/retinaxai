from __future__ import annotations
import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction
from app.models.prediction_explanation import ExplanationStatus, PredictionExplanation
from app.models.gradcam_explanation import GradCAMExplanation
from app.models.severity_report import SeverityReport
import structlog

from app.core.config import settings
from app.schemas.report_schema import ReportGenerateRequest

from app.api.v1.websockets import emit_xai_event
from app.explanations.utils import normalize_risk_level
from app.predictions.repository import PredictionRepository
from sqlalchemy.exc import IntegrityError

logger = structlog.get_logger(__name__)


class ExplanationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def store_xai_results(
        self,
        prediction_id: uuid.UUID,
        explanation_content: str | None = None,
        explanation_summary: str | None = None,
        explanation_model: str | None = None,
        gradcam_left_explanation: str | None = None,
        gradcam_right_explanation: str | None = None,
        severity_content: str | None = None,
        severity_summary: str | None = None,
        severity_risk_level: str = "moderate",
        severity_recommendations: list[str] | None = None,
    ) -> dict:
        prediction_repo = PredictionRepository(self.db)
        prediction = await prediction_repo.get_by_id(prediction_id)

        if not prediction:
            return {"status": "error", "message": "Prediction not found"}

        results: dict[str, list[str]] = {"stored": []}

        if explanation_content:
            exp = PredictionExplanation(
                id=uuid.uuid4(),
                prediction_id=prediction.id,
                content=explanation_content,
                summary=explanation_summary,
                model_used=explanation_model or "unknown",
                status="completed",
            )
            try:
                self.db.add(exp)
                await self.db.flush()
                results["stored"].append("prediction_explanation")
            except IntegrityError:
                await self.db.rollback()
                raise

            prediction.output_payload = prediction.output_payload or {}
            prediction.output_payload["explanation"] = explanation_content
            self.db.add(prediction)

        if gradcam_left_explanation or gradcam_right_explanation:
            gradcam_exp = GradCAMExplanation(
                id=uuid.uuid4(),
                prediction_id=prediction.id,
                left_eye_explanation=gradcam_left_explanation or "",
                right_eye_explanation=gradcam_right_explanation or "",
                highlighted_regions={},
                model_used=explanation_model or "unknown",
            )
            self.db.add(gradcam_exp)
            results["stored"].append("gradcam_explanation")

        if severity_content:
            risk_enum = normalize_risk_level(severity_risk_level)

            severity = SeverityReport(
                id=uuid.uuid4(),
                prediction_id=prediction.id,
                patient_id=prediction.patient_id,
                content=severity_content,
                summary=severity_summary,
                risk_level=risk_enum,
                recommendations=severity_recommendations or [],
                model_used=explanation_model or "unknown",
            )
            self.db.add(severity)
            results["stored"].append("severity_report")

            prediction.output_payload = prediction.output_payload or {}
            prediction.output_payload["severity_summary"] = severity_summary
            prediction.output_payload["severity_risk_level"] = risk_enum.value
            prediction.output_payload["severity_recommendations"] = (
                severity_recommendations or []
            )
            self.db.add(prediction)

        await self.db.commit()

        try:
            asyncio.create_task(
                emit_xai_event(
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

        return {"status": "ok", "prediction_id": str(prediction_id), "results": results}

    async def trigger_xai_for_prediction(
        self,
        prediction: Prediction,
        patient_data: dict,
    ) -> dict:
        """Trigger XAI pipeline after prediction completes."""
        import httpx

        from app.models.prediction import PredictionStatus

        output_payload = prediction.output_payload or {}
        dr_grade = output_payload.get("combined_grade", 0)
        confidence = prediction.confidence_score or 0.0
        gradcam_left = output_payload.get("gradcam_left")
        gradcam_right = output_payload.get("gradcam_right")
        gradcam_left_regions = output_payload.get("gradcam_left_regions", [])
        gradcam_right_regions = output_payload.get("gradcam_right_regions", [])

        left_region_names = [
            r.get("name")
            for r in gradcam_left_regions
            if isinstance(r, dict) and r.get("name")
        ]
        right_region_names = [
            r.get("name")
            for r in gradcam_right_regions
            if isinstance(r, dict) and r.get("name")
        ]
        left_regions_full = [
            r for r in gradcam_left_regions if isinstance(r, dict) and r.get("name")
        ]
        right_regions_full = [
            r for r in gradcam_right_regions if isinstance(r, dict) and r.get("name")
        ]

        results = {
            "prediction_explanation": None,
            "gradcam_explanation": None,
            "severity_report": None,
        }
        xai_failed = False

        gradcam_regions_full = {
            "left_eye": left_regions_full,
            "right_eye": right_regions_full,
        }
        gradcam_regions = {
            "left_eye": left_region_names,
            "right_eye": right_region_names,
        }

        llm_base_url = settings.LLM_SERVICE_URL
        async with httpx.AsyncClient(timeout=60.0) as client:
            if dr_grade is not None and dr_grade != "Unknown":
                dr_grade_value = (
                    str(dr_grade) if isinstance(dr_grade, int) else dr_grade
                )
                try:
                    resp = await client.post(
                        f"{llm_base_url}/api/xai/explain",
                        json={
                            "prediction_id": str(prediction.id),
                            "dr_grade": dr_grade_value,
                            "confidence": confidence,
                            "gradcam_regions": gradcam_regions_full,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data.get("content", "")

                        exp = PredictionExplanation(
                            id=uuid.uuid4(),
                            prediction_id=prediction.id,
                            content=content,
                            summary=data.get("summary"),
                            model_used=data.get("model_used", "unknown"),
                            status=ExplanationStatus.COMPLETED,
                        )
                        self.db.add(exp)
                        results["prediction_explanation"] = exp
                        logger.info(
                            f"[EXPLAIN SERVICE] Prediction explanation created for {prediction.id} (with SHAP)"
                        )
                except Exception as e:
                    xai_failed = True
                    logger.warning(f"[EXPLAIN SERVICE] XAI explain failed: {e}")

            if left_region_names or right_region_names:
                try:
                    resp = await client.post(
                        f"{llm_base_url}/api/xai/gradcam",
                        json={
                            "prediction_id": str(prediction.id),
                            "left_eye_regions": left_regions_full,
                            "right_eye_regions": right_regions_full,
                            "dr_grade": str(dr_grade) if dr_grade is not None else None,
                            "confidence": confidence,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        gradcam_exp = GradCAMExplanation(
                            id=uuid.uuid4(),
                            prediction_id=prediction.id,
                            left_eye_explanation=data.get("left_eye_explanation", ""),
                            right_eye_explanation=data.get("right_eye_explanation", ""),
                            highlighted_regions=data.get("highlighted_regions", {}),
                            model_used=data.get("model_used", "unknown"),
                        )
                        self.db.add(gradcam_exp)
                        results["gradcam_explanation"] = gradcam_exp
                        logger.info(
                            f"[EXPLAIN SERVICE] GradCAM explanation created for {prediction.id}"
                        )
                except Exception as e:
                    xai_failed = True
                    logger.warning(f"[EXPLAIN SERVICE] XAI gradcam failed: {e}")

            risk_factors = prediction.input_payload.get("risk_factors", [])
            dr_grade_value = str(dr_grade) if isinstance(dr_grade, int) else dr_grade
            try:
                resp = await client.post(
                    f"{llm_base_url}/api/xai/severity",
                    json={
                        "prediction_id": str(prediction.id),
                        "patient_data": patient_data,
                        "dr_grade": dr_grade_value,
                        "risk_factors": risk_factors,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    risk_level = normalize_risk_level(
                        data.get("risk_level", "moderate")
                    )
                    severity = SeverityReport(
                        id=uuid.uuid4(),
                        prediction_id=prediction.id,
                        patient_id=prediction.patient_id,
                        content=data.get("content", ""),
                        summary=data.get("summary"),
                        risk_level=risk_level,
                        recommendations=data.get("recommendations", []),
                        model_used=data.get("model_used", "unknown"),
                    )
                    self.db.add(severity)
                    results["severity_report"] = severity
                    logger.info(
                        f"[EXPLAIN SERVICE] Severity report created for {prediction.id}"
                    )
            except Exception as e:
                xai_failed = True
                logger.warning(f"[EXPLAIN SERVICE] XAI severity failed: {e}")

        await self.db.commit()

        if results["prediction_explanation"]:
            prediction.output_payload["explanation"] = results[
                "prediction_explanation"
            ].content
            await self.db.commit()

        try:
            from app.api.v1.websockets import emit_xai_event

            if results["prediction_explanation"]:
                asyncio.create_task(
                    emit_xai_event(
                        event_type="xai.explanation_ready",
                        prediction_id=str(prediction.id),
                        patient_id=str(prediction.patient_id),
                        status="completed",
                        progress=100,
                        message="Explanation generated successfully",
                        explanation_id=str(results["prediction_explanation"].id),
                        content=results["prediction_explanation"].content,
                        summary=results["prediction_explanation"].summary,
                    )
                )

            if results["gradcam_explanation"]:
                asyncio.create_task(
                    emit_xai_event(
                        event_type="xai.gradcam_ready",
                        prediction_id=str(prediction.id),
                        patient_id=str(prediction.patient_id),
                        status="completed",
                        progress=100,
                        message="GradCAM analysis complete",
                        explanation_id=str(results["gradcam_explanation"].id),
                        details={
                            "left_eye": results[
                                "gradcam_explanation"
                            ].left_eye_explanation,
                            "right_eye": results[
                                "gradcam_explanation"
                            ].right_eye_explanation,
                        },
                    )
                )

            if results["severity_report"]:
                asyncio.create_task(
                    emit_xai_event(
                        event_type="xai.severity_ready",
                        prediction_id=str(prediction.id),
                        patient_id=str(prediction.patient_id),
                        status="completed",
                        progress=100,
                        message=f"Risk assessment complete: {results['severity_report'].risk_level.value}",
                        explanation_id=str(results["severity_report"].id),
                        content=results["severity_report"].content,
                        summary=results["severity_report"].summary,
                        details={
                            "risk_level": results["severity_report"].risk_level.value,
                            "recommendations": results[
                                "severity_report"
                            ].recommendations,
                        },
                    )
                )

            logger.info(
                f"[EXPLAIN SERVICE] Emitted XAI WebSocket events for prediction {prediction.id}"
            )
        except Exception as ws_error:
            logger.warning(
                f"[EXPLAIN SERVICE] Failed to emit XAI WebSocket events: {ws_error}"
            )

        if not xai_failed:
            try:
                from app.reports.service import ReportService

                report_data = ReportGenerateRequest(
                    prediction_id=prediction.id,
                    report_type="prediction",
                )
                report_service = ReportService(self.db)
                await report_service.generate(report_data, prediction.requested_by)
                logger.info(
                    f"[EXPLAIN SERVICE] Report generation triggered for prediction {prediction.id}"
                )
            except Exception as report_error:
                logger.warning(
                    f"[EXPLAIN SERVICE] Failed to trigger report generation: {report_error}"
                )

        if xai_failed:
            prediction.status = PredictionStatus.PARTIAL
            await self.db.commit()
            logger.info(
                f"[EXPLAIN SERVICE] Prediction {prediction.id} marked as PARTIAL due to XAI failure"
            )

        return results
