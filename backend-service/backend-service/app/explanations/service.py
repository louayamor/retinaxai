from __future__ import annotations
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
from app.models.prediction import PredictionStatus
from app.predictions.repository import PredictionRepository
from app.reports.service import ReportService
from app.services.task_tracker import bg_tasks
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
            task = bg_tasks.create_task(
                emit_xai_event(
                    event_type="xai.explanation_ready",
                    prediction_id=str(prediction.id),
                    patient_id=str(prediction.patient_id),
                    status=ExplanationStatus.COMPLETED,
                    progress=100,
                    message="XAI explanation stored",
                ),
                name="emit_store_xai",
            )
            task.add_done_callback(
                lambda t: logger.warning("emit_xai_failed", error=str(t.exception()))
                if t.exception()
                else None
            )
        except RuntimeError as e:
            logger.warning("emit_xai_create_task_failed", error=str(e))

        return {"status": "ok", "prediction_id": str(prediction_id), "results": results}

    async def trigger_xai_for_prediction(
        self,
        prediction: Prediction,
        patient_data: dict,
    ) -> dict:
        import httpx

        from app.schemas.xai_schema import (
            XAIExplainResponse,
            XAIGradCAMResponse,
            XAISeverityResponse,
        )

        output_payload = prediction.output_payload or {}
        dr_grade = output_payload.get("combined_grade", 0)
        confidence = prediction.confidence_score or 0.0
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

        gradcam_regions_full: dict[str, list[dict]] = {
            "left_eye": left_regions_full,
            "right_eye": right_regions_full,
        }

        llm_base_url = settings.LLM_SERVICE_URL
        headers = {"Content-Type": "application/json"}
        if settings.LLM_SERVICE_API_KEY:
            headers["X-API-Key"] = settings.LLM_SERVICE_API_KEY

        async with httpx.AsyncClient(timeout=60.0) as client:
            if dr_grade is not None and dr_grade != "Unknown":
                dr_grade_value = (
                    str(dr_grade) if isinstance(dr_grade, int) else dr_grade
                )
                try:
                    # Include lesion counts/clusters if available
                    lesions_left = output_payload.get("lesions_left") or output_payload.get("lesions")
                    lesions_right = output_payload.get("lesions_right")
                    lesion_clusters_left = output_payload.get("lesion_clusters_left") or output_payload.get("lesion_clusters")
                    lesion_clusters_right = output_payload.get("lesion_clusters_right")

                    resp = await client.post(
                        f"{llm_base_url}/api/xai/explain",
                        json={
                            "prediction_id": str(prediction.id),
                            "dr_grade": dr_grade_value,
                            "confidence": confidence,
                            "gradcam_regions": gradcam_regions_full,
                            "lesions_left": lesions_left,
                            "lesions_right": lesions_right,
                            "lesion_clusters_left": lesion_clusters_left,
                            "lesion_clusters_right": lesion_clusters_right,
                        },
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        parsed = XAIExplainResponse.model_validate(resp.json())
                        exp = PredictionExplanation(
                            id=uuid.uuid4(),
                            prediction_id=prediction.id,
                            content=parsed.content,
                            summary=parsed.summary,
                            model_used=parsed.model_used,
                            status=ExplanationStatus.COMPLETED,
                        )
                        self.db.add(exp)
                        results["prediction_explanation"] = exp
                        logger.info(
                            f"[EXPLAIN SERVICE] Prediction explanation created for {prediction.id}"
                        )
                except Exception as e:
                    xai_failed = True
                    logger.warning("xai_explain_failed", error=str(e))

            if left_region_names or right_region_names:
                try:
                    # Include lesion clusters in gradcam payload if present
                    resp = await client.post(
                        f"{llm_base_url}/api/xai/gradcam",
                        json={
                            "prediction_id": str(prediction.id),
                            "left_eye_regions": left_regions_full,
                            "right_eye_regions": right_regions_full,
                            "dr_grade": str(dr_grade) if dr_grade is not None else None,
                            "confidence": confidence,
                            "lesion_clusters_left": output_payload.get("lesion_clusters_left") or output_payload.get("lesion_clusters"),
                            "lesion_clusters_right": output_payload.get("lesion_clusters_right"),
                        },
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        parsed = XAIGradCAMResponse.model_validate(resp.json())
                        gradcam_exp = GradCAMExplanation(
                            id=uuid.uuid4(),
                            prediction_id=prediction.id,
                            left_eye_explanation=parsed.left_eye_explanation,
                            right_eye_explanation=parsed.right_eye_explanation,
                            highlighted_regions=parsed.highlighted_regions,
                            model_used=parsed.model_used,
                        )
                        self.db.add(gradcam_exp)
                        results["gradcam_explanation"] = gradcam_exp
                        logger.info(
                            f"[EXPLAIN SERVICE] GradCAM explanation created for {prediction.id}"
                        )
                except Exception as e:
                    xai_failed = True
                    logger.warning("xai_gradcam_failed", error=str(e))

            risk_factors = prediction.input_payload.get("risk_factors", [])
            dr_grade_value = str(dr_grade) if isinstance(dr_grade, int) else dr_grade
            try:
                # Include lesion counts in severity payload to help risk estimation
                resp = await client.post(
                    f"{llm_base_url}/api/xai/severity",
                    json={
                        "prediction_id": str(prediction.id),
                        "patient_data": patient_data,
                        "dr_grade": dr_grade_value,
                        "risk_factors": risk_factors,
                        "lesions_left": output_payload.get("lesions_left") or output_payload.get("lesions"),
                        "lesions_right": output_payload.get("lesions_right"),
                    },
                    headers=headers,
                )
                if resp.status_code == 200:
                    parsed = XAISeverityResponse.model_validate(resp.json())
                    risk_level = normalize_risk_level(parsed.risk_level)
                    severity = SeverityReport(
                        id=uuid.uuid4(),
                        prediction_id=prediction.id,
                        patient_id=prediction.patient_id,
                        content=parsed.content,
                        summary=parsed.summary,
                        risk_level=risk_level,
                        recommendations=parsed.recommendations,
                        model_used=parsed.model_used,
                    )
                    self.db.add(severity)
                    results["severity_report"] = severity
                    logger.info(
                        f"[EXPLAIN SERVICE] Severity report created for {prediction.id}"
                    )
            except Exception as e:
                xai_failed = True
                logger.warning("xai_severity_failed", error=str(e))

        await self.db.commit()

        if results["prediction_explanation"]:
            prediction.output_payload["explanation"] = results[
                "prediction_explanation"
            ].content
            await self.db.commit()

        try:
            if results["prediction_explanation"]:
                bg_tasks.create_task(
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
                bg_tasks.create_task(
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
                bg_tasks.create_task(
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
