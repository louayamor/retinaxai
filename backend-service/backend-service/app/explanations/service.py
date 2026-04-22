import asyncio
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction
from app.models.prediction_explanation import ExplanationStatus, PredictionExplanation
from app.models.gradcam_explanation import GradCAMExplanation
from app.models.severity_report import RiskLevel, SeverityReport
from app.schemas.report_schema import ReportGenerateRequest

logger = logging.getLogger(__name__)

LLM_SERVICE_URL = "http://localhost:8002"

RISK_LEVEL_ALIASES: dict[str, str] = {
    "very_high": "severe",
}


def _normalize_risk_level(risk_level: str | None) -> RiskLevel:
    if not risk_level:
        return RiskLevel.MODERATE
    normalized = RISK_LEVEL_ALIASES.get(risk_level.lower(), risk_level.lower())
    try:
        return RiskLevel(normalized)
    except ValueError:
        return RiskLevel.MODERATE


class ExplanationService:
    def __init__(self, db: AsyncSession):
        self.db = db

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
        vascular_biomarkers = patient_data.get("vascular_biomarkers", {})

        left_region_names = [
            r.get("name") for r in gradcam_left_regions if isinstance(r, dict) and r.get("name")
        ]
        right_region_names = [
            r.get("name") for r in gradcam_right_regions if isinstance(r, dict) and r.get("name")
        ]

        results = {
            "prediction_explanation": None,
            "gradcam_explanation": None,
            "severity_report": None,
        }
        xai_failed = False
        shap_values = None

        gradcam_regions = {
            "left_eye": left_region_names,
            "right_eye": right_region_names,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            if dr_grade is not None and dr_grade != "Unknown":
                dr_grade_value = str(dr_grade) if isinstance(dr_grade, int) else dr_grade
                try:
                    resp = await client.post(
                        f"{LLM_SERVICE_URL}/api/xai/explain",
                        json={
                            "prediction_id": str(prediction.id),
                            "dr_grade": dr_grade_value,
                            "confidence": confidence,
                            "clinical_features": prediction.input_payload,
                            "gradcam_regions": gradcam_regions,
                            "vascular_biomarkers": vascular_biomarkers,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        shap_values = data.get("shap_values")
                        content = data.get("content", "")
                        if shap_values:
                            top_features = [
                                f["name"]
                                for f in shap_values.get("top_positive", [])[:3]
                            ]
                            content = f"{content}\n\n[SHAP Analysis: Top features include {', '.join(top_features)}]"

                        exp = PredictionExplanation(
                            id=uuid.uuid4(),
                            prediction_id=prediction.id,
                            content=content,
                            summary=data.get("summary"),
                            model_used=data.get("model_used", "unknown"),
                            status=ExplanationStatus.COMPLETED,
                            shap_values=shap_values,
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
                        f"{LLM_SERVICE_URL}/api/xai/gradcam",
                        json={
                            "prediction_id": str(prediction.id),
                            "left_eye_regions": left_region_names,
                            "right_eye_regions": right_region_names,
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
                    f"{LLM_SERVICE_URL}/api/xai/severity",
                    json={
                        "prediction_id": str(prediction.id),
                        "patient_data": patient_data,
                        "dr_grade": dr_grade_value,
                        "risk_factors": risk_factors,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    risk_level = _normalize_risk_level(data.get("risk_level", "moderate"))
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
            if shap_values:
                prediction.output_payload["shap_values"] = shap_values
            await self.db.commit()

        try:
            from app.websockets.manager import get_socket_manager

            socket_manager = get_socket_manager()

            if results["prediction_explanation"]:
                asyncio.create_task(
                    socket_manager.emit_xai_event(
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
                    socket_manager.emit_xai_event(
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
                    socket_manager.emit_xai_event(
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
