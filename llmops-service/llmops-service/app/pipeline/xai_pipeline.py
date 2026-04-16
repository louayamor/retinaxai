from __future__ import annotations

import json

from loguru import logger

from app.core.config import settings
from app.llm.client import get_llm_client
from app.prompts.templates import REPORT_SYSTEM_PROMPT
from app.services.websocket_client import send_xai_event


XAI_EXPLANATION_SYSTEM_PROMPT = """Generate structured diabetic retinopathy explanations in JSON format.

Output JSON with these exact keys:
- diagnosis: {condition, severity, overall_grade (0-4), confidence (0.0-1.0), risk_level}
- clinical_findings: {left_eye: {grade, severity, confidence, description}, right_eye: {...}}
- feature_importance: {top_contributors: [{feature_name, contribution}], key_insights: []}
- clinical_context: {risk_factors: [], visual_indicators: [], recommendations: []}
- summary: 3-4 sentence clinical summary

DATA REQUIREMENTS:
- confidence: MUST be decimal like 0.79, NOT "79%" or 79
- grade: MUST be integer 0-4
- description: At least 2 sentences per eye with specific findings (microaneurysms, hemorrhages, etc.)

Output complete valid JSON only."""


XAI_SEVERITY_SYSTEM_PROMPT = """Generate structured severity reports in JSON format.

Output JSON with these exact keys:
- patient: {name, age, gender}
- diagnosis: {condition, dr_grade (0-4), severity_label, risk_level}
- clinical_assessment: {findings, visual_indicators: [], comparison_to_previous}
- risk_factors: []
- risk_stratification: {overall_risk, progression_risk, vision_loss_risk}
- recommendations: [{action, timeframe, rationale}]
- follow_up: {next_appointment, frequency, tests_required: []}
- summary: 4-5 sentence clinical summary

DATA REQUIREMENTS:
- findings: At least 3 sentences with specific observations
- recommendations: 3-5 items with urgency levels (immediate/urgent/routine)
- summary: Must complete the full clinical narrative

Output complete valid JSON only."""


class XAIPipeline:
    _instance: "XAIPipeline | None" = None

    def __new__(cls) -> "XAIPipeline":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        provider = (
            settings.llm_provider.value
            if hasattr(settings.llm_provider, "value")
            else str(settings.llm_provider)
        )
        token = settings.github_token if provider == "github" else settings.llm_api_key
        base_url = (
            settings.github_endpoint if provider == "github" else settings.llm_base_url
        )

        timeout = settings.timeout_seconds
        max_tokens = settings.max_tokens

        client_kwargs: dict[str, str | int] = {
            "model": settings.llm_model,
            "timeout_seconds": timeout,
            "max_tokens": max_tokens,
        }
        if provider == "github":
            client_kwargs["token"] = token if token is not None else ""
            client_kwargs["endpoint"] = base_url if base_url is not None else ""
        elif provider == "ollama":
            client_kwargs["base_url"] = (
                base_url if base_url is not None else settings.ollama_base_url
            )
        else:
            client_kwargs["token"] = token if token is not None else ""
            client_kwargs["base_url"] = base_url if base_url is not None else ""

        self.client = get_llm_client(provider, **client_kwargs)
        self._initialized = True
        logger.info("XAI Pipeline initialized")

    async def explain_prediction(
        self,
        prediction_id: str,
        dr_grade: str,
        confidence: float,
        clinical_features: dict | None = None,
    ) -> dict:
        """Generate natural language explanation of DR prediction using SHAP values."""
        await send_xai_event(
            event="xai.prediction",
            stage="prediction",
            status="started",
            progress=0,
            message="Generating prediction explanation with SHAP...",
            prediction_id=prediction_id,
        )

        shap_values = None
        shap_explanation = None

        if clinical_features:
            try:
                from app.services.shap_service import get_shap_service

                await send_xai_event(
                    event="xai.prediction",
                    stage="prediction",
                    status="progress",
                    progress=25,
                    message="Calculating SHAP feature contributions...",
                    prediction_id=prediction_id,
                )

                shap_service = get_shap_service()
                shap_explanation = shap_service.explain_prediction(
                    features=clinical_features,
                    pipeline="clinical",
                )
                shap_values = shap_explanation.to_dict()
                logger.info(f"SHAP explanation computed for prediction {prediction_id}")

            except Exception as shap_error:
                logger.warning(
                    f"SHAP calculation failed, continuing without it: {shap_error}"
                )
                shap_values = None
                shap_explanation = None

        await send_xai_event(
            event="xai.prediction",
            stage="prediction",
            status="progress",
            progress=50,
            message="Generating narrative explanation...",
            prediction_id=prediction_id,
        )

        try:
            prompt = self._build_prediction_prompt_with_shap(
                dr_grade, confidence, clinical_features, shap_values
            )
            response = self.client.generate(prompt)

            await send_xai_event(
                event="xai.prediction",
                stage="prediction",
                status="completed",
                progress=100,
                message="Prediction explanation generated",
                prediction_id=prediction_id,
                details={"dr_grade": dr_grade, "confidence": confidence},
            )

            result_details = {
                "dr_grade": dr_grade,
                "confidence": confidence,
                "content": response,
                "summary": response[:500],
            }
            if shap_values:
                result_details["shap_values"] = shap_values
                result_details["top_features"] = shap_values.get("top_positive", [])

            await send_xai_event(
                event="xai.explanation_ready",
                stage="prediction",
                status="completed",
                progress=100,
                message="Explanation ready",
                prediction_id=prediction_id,
                details=result_details,
            )

            return {
                "content": response,
                "summary": response[:500],
                "model_used": settings.llm_model,
                "status": "completed",
                "shap_values": shap_values,
            }
        except Exception as e:
            await send_xai_event(
                event="xai.prediction",
                stage="prediction",
                status="failed",
                progress=0,
                message=str(e),
                prediction_id=prediction_id,
                error=str(e),
            )
            raise

    async def explain_gradcam(
        self,
        prediction_id: str,
        left_eye_regions: list[str],
        right_eye_regions: list[str],
    ) -> dict:
        """Interpret highlighted regions in GradCAM heatmaps."""
        await send_xai_event(
            event="xai.gradcam",
            stage="gradcam",
            status="started",
            progress=0,
            message="Interpreting GradCAM regions...",
            prediction_id=prediction_id,
        )

        try:
            prompt = self._build_gradcam_prompt(left_eye_regions, right_eye_regions)
            response = self.client.generate(prompt)

            highlighted_regions = {
                "left_eye": left_eye_regions,
                "right_eye": right_eye_regions,
            }

            await send_xai_event(
                event="xai.gradcam",
                stage="gradcam",
                status="completed",
                progress=100,
                message="GradCAM interpretation complete",
                prediction_id=prediction_id,
                details={
                    "left_regions": len(left_eye_regions),
                    "right_regions": len(right_eye_regions),
                },
            )

            await send_xai_event(
                event="xai.gradcam_ready",
                stage="gradcam",
                status="completed",
                progress=100,
                message="GradCAM analysis ready",
                prediction_id=prediction_id,
                details={
                    "left_eye": response,
                    "right_eye": response,
                    "highlighted_regions": highlighted_regions,
                },
            )

            return {
                "left_eye_explanation": response,
                "right_eye_explanation": response,
                "highlighted_regions": highlighted_regions,
                "model_used": settings.llm_model,
            }
        except Exception as e:
            await send_xai_event(
                event="xai.gradcam",
                stage="gradcam",
                status="failed",
                progress=0,
                message=str(e),
                prediction_id=prediction_id,
                error=str(e),
            )
            raise

    async def generate_severity_report(
        self,
        prediction_id: str,
        patient_data: dict,
        dr_grade: str,
        risk_factors: list[str],
    ) -> dict:
        """Generate clinical severity report with risk level and recommendations."""
        await send_xai_event(
            event="xai.severity",
            stage="severity",
            status="started",
            progress=0,
            message="Generating severity report...",
            prediction_id=prediction_id,
        )

        try:
            prompt = self._build_severity_prompt(patient_data, dr_grade, risk_factors)
            response = self.client.generate(prompt)

            risk_level = self._determine_risk_level(dr_grade)
            recommendations = self._generate_recommendations(dr_grade, risk_factors)

            await send_xai_event(
                event="xai.severity",
                stage="severity",
                status="completed",
                progress=100,
                message="Severity report generated",
                prediction_id=prediction_id,
                details={"risk_level": risk_level},
            )

            await send_xai_event(
                event="xai.severity_ready",
                stage="severity",
                status="completed",
                progress=100,
                message=f"Severity report ready: {risk_level}",
                prediction_id=prediction_id,
                details={
                    "content": response,
                    "summary": response[:500],
                    "risk_level": risk_level,
                    "recommendations": recommendations,
                },
            )

            return {
                "content": response,
                "summary": response[:500],
                "risk_level": risk_level,
                "recommendations": recommendations,
                "model_used": settings.llm_model,
            }
        except Exception as e:
            await send_xai_event(
                event="xai.severity",
                stage="severity",
                status="failed",
                progress=0,
                message=str(e),
                prediction_id=prediction_id,
                error=str(e),
            )
            raise

    def _build_prediction_prompt(
        self,
        dr_grade: str,
        confidence: float,
        clinical_features: dict | None,
    ) -> str:
        grade_int = int(dr_grade) if dr_grade.isdigit() else 2
        grade_label = (
            ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][grade_int]
            if 0 <= grade_int <= 4
            else "Moderate"
        )

        clinical_context = ""
        if clinical_features:
            clinical_context = f"\nClinical features: {json.dumps(clinical_features)}"

        return f"""{XAI_EXPLANATION_SYSTEM_PROMPT}

PREDICTION DATA:
- DR Grade: {grade_int} ({grade_label})
- Confidence: {confidence:.2f}
{clinical_context}

Generate structured explanation as JSON."""

    def _build_prediction_prompt_with_shap(
        self,
        dr_grade: str,
        confidence: float,
        clinical_features: dict | None,
        shap_values: dict | None,
    ) -> str:
        shap_context = ""
        if shap_values:
            top_positive = shap_values.get("top_positive", [])
            top_negative = shap_values.get("top_negative", [])
            expected_value = shap_values.get("expected_value", 0)

            positive_features = (
                ", ".join(
                    [f"{f['name']} ({f['contribution']:.3f})" for f in top_positive[:3]]
                )
                if top_positive
                else "None"
            )
            negative_features = (
                ", ".join(
                    [f"{f['name']} ({f['contribution']:.3f})" for f in top_negative[:3]]
                )
                if top_negative
                else "None"
            )

            shap_context = f"""
SHAP Feature Analysis:
- Base value (expected): {expected_value:.3f}
- Top positive contributing features: {positive_features}
- Top negative contributing features: {negative_features}
"""

        clinical_context = ""
        if clinical_features:
            clinical_context = f"\nClinical Features: {clinical_features}"

        prompt = f"""You are a medical AI assistant explaining diabetic retinopathy (DR) prediction results.

Explain this prediction in patient-friendly terms addressing these key areas:

1. DIAGNOSIS: The DR grade is {dr_grade} with {confidence:.1%} confidence.

2. FEATURE CONTRIBUTIONS:{shap_context}

3. CLINICAL CONTEXT:{clinical_context}

Please provide:
- A clear explanation of what this diagnosis means for the patient
- How the key clinical features influenced the prediction
- Recommended next steps and follow-up actions
- Any warning signs the patient should watch for

Keep the explanation professional but accessible to a non-medical patient."""

        return f"{REPORT_SYSTEM_PROMPT}\n\n{prompt}"

    def _build_gradcam_prompt(
        self,
        left_regions: list[str],
        right_regions: list[str],
    ) -> str:
        return f"""Interpret these highlighted regions from GradCAM heatmaps:

Left Eye: {", ".join(left_regions)}
Right Eye: {", ".join(right_regions)}

Explain what these regions indicate for DR diagnosis."""

    def _build_severity_prompt(
        self,
        patient_data: dict,
        dr_grade: str,
        risk_factors: list[str],
    ) -> str:
        grade_int = int(dr_grade) if dr_grade.isdigit() else 2
        grade_label = (
            ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][grade_int]
            if 0 <= grade_int <= 4
            else "Moderate"
        )

        patient_info = f"Name: {patient_data.get('name', 'Unknown')}, Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}"
        risk_factors_str = ", ".join(risk_factors) if risk_factors else "None provided"

        return f"""{XAI_SEVERITY_SYSTEM_PROMPT}

PATIENT DATA:
{patient_info}

DIAGNOSIS:
- DR Grade: {grade_int} ({grade_label})
- Risk Factors: {risk_factors_str}

Generate structured severity report as JSON."""

    def _determine_risk_level(self, dr_grade: str) -> str:
        mapping = {
            "No DR": "low",
            "Mild": "low",
            "Moderate": "moderate",
            "Severe": "high",
            "Proliferative DR": "severe",
        }
        return mapping.get(dr_grade, "moderate")

    def _generate_recommendations(
        self, dr_grade: str, risk_factors: list[str]
    ) -> list[str]:
        recommendations = []

        if dr_grade in ("Mild", "No DR"):
            recommendations.extend(
                [
                    "Annual retinal screening",
                    "Maintain blood sugar control",
                ]
            )
        elif dr_grade == "Moderate":
            recommendations.extend(
                [
                    "Screen every 6 months",
                    "Consider laser therapy consultation",
                ]
            )
        elif dr_grade == "Severe":
            recommendations.extend(
                [
                    "Immediate ophthalmology referral",
                    "Consider laser therapy",
                ]
            )
        elif dr_grade == "Proliferative DR":
            recommendations.extend(
                [
                    "Urgent vitrectomy consultation",
                    "Surgical intervention required",
                ]
            )

        if "hypertension" in risk_factors:
            recommendations.append("Blood pressure management")
        if "gestational_diabetes" in risk_factors:
            recommendations.append("Post-partum follow-up")

        return recommendations


def get_xai_pipeline() -> XAIPipeline:
    return XAIPipeline()
