from __future__ import annotations

import json

from loguru import logger

from app.core.config import settings
from app.llm.client import get_llm_client
from app.prompts.templates import (
    GRADCAM_SYSTEM_PROMPT,
    GRADCAM_USER_PROMPT,
    REPORT_SYSTEM_PROMPT,
)
from app.services.shap_service import ShapService
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


class InvalidGradeError(ValueError):
    """Raised when dr_grade is not a valid integer 0-4."""

    pass


def _validate_dr_grade(dr_grade: str | int) -> int:
    """Validate and convert dr_grade to an integer 0-4.

    Args:
        dr_grade: DR grade as string or int.

    Returns:
        int: Validated grade integer (0-4).

    Raises:
        InvalidGradeError: If grade is not a valid integer in range 0-4.
    """
    if isinstance(dr_grade, int):
        grade_int = dr_grade
    elif isinstance(dr_grade, str) and dr_grade.isdigit():
        grade_int = int(dr_grade)
    else:
        raise InvalidGradeError(
            f"dr_grade must be an integer or numeric string 0-4, got {type(dr_grade).__name__}: {dr_grade!r}"
        )

    if not 0 <= grade_int <= 4:
        raise InvalidGradeError(f"dr_grade must be between 0 and 4, got {grade_int}")

    return grade_int


# Bug 7 fix: Add grade-to-risk mappings
_GRADE_INT_TO_RISK = {0: "low", 1: "low", 2: "moderate", 3: "high", 4: "severe"}
_GRADE_LABEL_TO_RISK = {
    "No DR": "low",
    "Mild": "low",
    "Moderate": "moderate",
    "Severe": "high",
    "Proliferative DR": "severe",
}


class XAIPipeline:
    """Pipeline for XAI explanations using LLM."""

    def __init__(self) -> None:
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
        logger.info("XAI Pipeline initialized")

    async def explain_prediction(
        self,
        prediction_id: str,
        dr_grade: str,
        confidence: float,
        clinical_features: dict | None = None,
        gradcam_regions: dict | None = None,
        vascular_biomarkers: dict | None = None,
    ) -> dict:
        """Generate natural language explanation of DR prediction using SHAP values or GradCAM regions."""
        await send_xai_event(
            event="xai.prediction",
            stage="prediction",
            status="started",
            progress=0,
            message="Generating prediction explanation...",
            prediction_id=prediction_id,
        )

        shap_values = None
        shap_explanation = None
        imaging_explanation = None

        if gradcam_regions and (
            gradcam_regions.get("left_eye") or gradcam_regions.get("right_eye")
        ):
            try:
                from app.services.shap_service import get_shap_service

                await send_xai_event(
                    event="xai.prediction",
                    stage="prediction",
                    status="progress",
                    progress=25,
                    message="Calculating region-based feature contributions from GradCAM...",
                    prediction_id=prediction_id,
                )

                shap_service = get_shap_service()
                grade_int = _validate_dr_grade(dr_grade)
                imaging_explanation = shap_service.explain_imaging_prediction(
                    regions=gradcam_regions,
                    prediction_grade=grade_int,
                    confidence=confidence,
                )
                shap_values = imaging_explanation.to_dict()
                logger.info(
                    f"Imaging explanation computed for prediction {prediction_id} using GradCAM regions"
                )

            except Exception as imaging_error:
                logger.warning(
                    f"Imaging explanation calculation failed, continuing without it: {imaging_error}"
                )
                shap_values = None
                imaging_explanation = None

        elif clinical_features:
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
                shap_explanation = await shap_service.explain_prediction(
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
            if imaging_explanation:
                prompt = self._build_imaging_prompt_with_regions(
                    dr_grade,
                    confidence,
                    gradcam_regions,
                    shap_values,
                    vascular_biomarkers,
                )
            else:
                prompt = self._build_prediction_prompt_with_shap(
                    dr_grade,
                    confidence,
                    clinical_features,
                    shap_values,
                    vascular_biomarkers,
                )
            logger.info(f"Starting LLM generation for prediction {prediction_id}")
            response = await self.client.generate(prompt)
            logger.info(f"LLM generation completed for prediction {prediction_id}")

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
                if imaging_explanation:
                    result_details["explanation_type"] = "gradcam_regions"
                    result_details["top_regions"] = shap_values.get("top_regions", [])
                else:
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
                "explanation_type": "gradcam_regions"
                if imaging_explanation
                else "clinical_shap",
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

    def _build_imaging_prompt_with_regions(
        self,
        dr_grade: str,
        confidence: float,
        gradcam_regions: dict | None,
        shap_values: dict | None,
        vascular_biomarkers: dict | None,
    ) -> str:
        """Build prompt for imaging-based explanation using GradCAM regions.

        Includes per-eye clinical analysis with specific DR pathology terminology.
        """
        grade_int = _validate_dr_grade(dr_grade)
        grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
            grade_int
        ]
        risk_level = _GRADE_INT_TO_RISK.get(grade_int, "moderate")

        # Build per-eye region analysis with clinical context
        left_eye_analysis = ""
        right_eye_analysis = ""

        if gradcam_regions:
            left_regions = gradcam_regions.get("left_eye", [])
            right_regions = gradcam_regions.get("right_eye", [])

            if left_regions:
                left_clinical = []
                for region in left_regions:
                    clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                        region,
                        {
                            "significance": "Retinal region",
                            "high_contribution": "DR-related changes",
                        },
                    )
                    pathology = self._get_pathology_for_grade(grade_int, region)
                    left_clinical.append(
                        f"**{region}**: {clinical_info['significance']}. "
                        f"At grade {grade_int} ({grade_label}), expected findings: {pathology}. "
                        f"Clinical note: {clinical_info.get('high_contribution', 'Monitor for progression')}."
                    )
                left_eye_analysis = "\n".join(left_clinical)

            if right_regions:
                right_clinical = []
                for region in right_regions:
                    clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                        region,
                        {
                            "significance": "Retinal region",
                            "high_contribution": "DR-related changes",
                        },
                    )
                    pathology = self._get_pathology_for_grade(grade_int, region)
                    right_clinical.append(
                        f"**{region}**: {clinical_info['significance']}. "
                        f"At grade {grade_int} ({grade_label}), expected findings: {pathology}. "
                        f"Clinical note: {clinical_info.get('high_contribution', 'Monitor for progression')}."
                    )
                right_eye_analysis = "\n".join(right_clinical)

        shap_context = ""
        if shap_values:
            top_regions = shap_values.get("top_regions", [])
            if top_regions:
                regions_str = ", ".join(
                    [
                        f"{r['name']} (contribution: {r['contribution']:.3f})"
                        for r in top_regions[:3]
                    ]
                )
                shap_context = f"""
Region Importance Analysis (GradCAM activation strength):
{regions_str}
"""

        biomarker_context = ""
        if vascular_biomarkers:
            biomarker_context = f"""
Vascular Biomarkers (quantitative analysis):
{json.dumps(vascular_biomarkers, indent=2)}
"""

        prompt = f"""You are a retinal specialist explaining diabetic retinopathy (DR) imaging findings.

PATIENT STATUS:
- DR Grade: {grade_int} ({grade_label})
- Model Confidence: {confidence:.1%}
- Risk Level: {risk_level}

GRADCAM HIGHLIGHTED REGIONS - PER-EYE ANALYSIS:

LEFT EYE (OS):
{left_eye_analysis if left_eye_analysis else "No regions highlighted."}

RIGHT EYE (OD):
{right_eye_analysis if right_eye_analysis else "No regions highlighted."}

{shap_context}
{biomarker_context}

CLINICAL INTERPRETATION REQUIREMENTS:
1. Explain what the highlighted regions indicate about DR pathology in each eye
2. Correlate specific findings (microaneurysms, hemorrhages, exudates, neovascularization) with the DR grade
3. Describe the anatomical significance: why these regions matter for vision
4. Integrate vascular biomarkers if provided
5. Provide specific follow-up recommendations based on the grade and regions involved

Use precise clinical terminology. Avoid generic phrases like "the model focuses on."
Instead use: "findings consistent with," "pathology characteristic of," "changes suggestive of."

Write as a retinal specialist documenting in a clinical report."""

        return f"{REPORT_SYSTEM_PROMPT}\n\n{prompt}"

    def _get_pathology_for_grade(self, grade_int: int, region: str) -> str:
        """Return expected DR pathology for a given grade and region.

        Args:
            grade_int: DR grade (0-4).
            region: Anatomical region name.

        Returns:
            String describing expected pathology.
        """
        # Central regions (macula, fovea) - vision-threatening findings
        central_regions = [
            "fovea_centralis",
            "macula_center",
            "perifovea",
            "superior_macula",
            "inferior_macula",
            "posterior_pole",
        ]

        # Vascular regions - vascular findings
        vascular_regions = [
            "temporal_arcade",
            "superior_temporal_arcade",
            "inferior_temporal_arcade",
            "superior_arcade",
            "inferior_arcade",
        ]

        # Peripheral regions - neovascularization in PDR
        peripheral_regions = [
            "nasal_periphery",
            "temporal_periphery",
            "superior_periphery",
            "inferior_periphery",
            "superior_temporal_periphery",
            "inferior_temporal_periphery",
            "superior_nasal_periphery",
            "inferior_nasal_periphery",
            "mid_periphery",
        ]

        if grade_int == 0:
            return "No DR findings expected"
        elif grade_int == 1:  # Mild NPDR
            if region in central_regions:
                return "occasional microaneurysms"
            elif region in vascular_regions:
                return "mild venous dilation"
            else:
                return "minimal retinal changes"
        elif grade_int == 2:  # Moderate NPDR
            if region in central_regions:
                return "microaneurysms, dot-blot hemorrhages, possible hard exudates"
            elif region in vascular_regions:
                return "venous beading, intraretinal microvascular abnormalities (IRMA)"
            else:
                return "scattered hemorrhages and microaneurysms"
        elif grade_int == 3:  # Severe NPDR
            if region in central_regions:
                return "extensive hemorrhages, cotton wool spots, hard exudates threatening fovea"
            elif region in vascular_regions:
                return "severe venous beading, prominent IRMA, pre-retinal hemorrhages"
            elif region in peripheral_regions:
                return "extensive dot-blot hemorrhages in all 4 quadrants"
            else:
                return "severe non-proliferative changes"
        else:  # grade_int == 4, Proliferative DR
            if region in peripheral_regions:
                return "neovascularization elsewhere (NVE), fibrovascular proliferation"
            elif region in vascular_regions:
                return "neovascularization of disk (NVD) or elsewhere, vitreous hemorrhage risk"
            elif region in central_regions:
                return "tractional retinal detachment risk, neovascularization threatening macula"
            else:
                return "proliferative changes with neovascularization"

        return "DR-related retinal changes"

    async def explain_gradcam(
        self,
        prediction_id: str,
        left_eye_regions: list[str],
        right_eye_regions: list[str],
        dr_grade: str | int | None = None,
        confidence: float | None = None,
    ) -> dict:
        """Interpret highlighted regions in GradCAM heatmaps with clinical specificity.

        Args:
            prediction_id: Unique identifier for the prediction.
            left_eye_regions: List of anatomical region names for left eye.
            right_eye_regions: List of anatomical region names for right eye.
            dr_grade: Optional DR grade (0-4) for clinical context.
            confidence: Optional model confidence (0-1) for clinical context.

        Returns:
            dict with left_eye_explanation and right_eye_explanation (distinct per eye).
        """
        await send_xai_event(
            event="xai.gradcam",
            stage="gradcam",
            status="started",
            progress=0,
            message="Interpreting GradCAM regions...",
            prediction_id=prediction_id,
        )

        try:
            # Default grade and confidence if not provided
            grade_int = _validate_dr_grade(dr_grade) if dr_grade is not None else 2
            grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
                grade_int
            ]
            conf = confidence if confidence is not None else 0.75
            risk_level = _GRADE_INT_TO_RISK.get(grade_int, "moderate")

            # Generate per-eye explanations with clinical context
            left_prompt = self._build_gradcam_prompt_per_eye(
                regions=left_eye_regions,
                eye_name="Left Eye (OS)",
                grade_int=grade_int,
                grade_label=grade_label,
                confidence=conf,
                risk_level=risk_level,
            )
            right_prompt = self._build_gradcam_prompt_per_eye(
                regions=right_eye_regions,
                eye_name="Right Eye (OD)",
                grade_int=grade_int,
                grade_label=grade_label,
                confidence=conf,
                risk_level=risk_level,
            )

            left_response = await self.client.generate(
                left_prompt, system_prompt=GRADCAM_SYSTEM_PROMPT
            )
            right_response = await self.client.generate(
                right_prompt, system_prompt=GRADCAM_SYSTEM_PROMPT
            )

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
                    "dr_grade": grade_label,
                    "confidence": conf,
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
                    "left_eye": left_response,
                    "right_eye": right_response,
                    "highlighted_regions": highlighted_regions,
                    "dr_grade": grade_label,
                    "confidence": conf,
                },
            )

            return {
                "left_eye_explanation": left_response,
                "right_eye_explanation": right_response,
                "highlighted_regions": highlighted_regions,
                "dr_grade": grade_label,
                "confidence": conf,
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

    def _build_gradcam_prompt_per_eye(
        self,
        regions: list[str],
        eye_name: str,
        grade_int: int,
        grade_label: str,
        confidence: float,
        risk_level: str,
    ) -> str:
        """Build clinical prompt for a single eye's GradCAM regions.

        Args:
            regions: List of anatomical region names.
            eye_name: "Left Eye (OS)" or "Right Eye (OD)".
            grade_int: DR grade integer (0-4).
            grade_label: DR grade label (e.g., "Moderate").
            confidence: Model confidence (0-1).
            risk_level: Risk level string (low/moderate/high/severe).

        Returns:
            Formatted prompt string with clinical context.
        """
        # Build region list with clinical relevance
        regions_with_context = []
        for region in regions:
            clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                region,
                {
                    "significance": "Retinal region",
                    "high_contribution": "Model activation in this region",
                    "moderate_contribution": "Regional changes detected",
                },
            )
            regions_with_context.append(
                f"- {region}: {clinical_info['significance']}. "
                f"At grade {grade_int} ({grade_label}), findings may include: "
                f"{clinical_info.get('high_contribution', 'DR-related changes')}."
            )

        regions_text = (
            "\n".join(regions_with_context)
            if regions_with_context
            else "No regions highlighted."
        )

        return GRADCAM_USER_PROMPT.format(
            grade_int=str(grade_int),
            grade_label=grade_label,
            confidence=confidence,
            risk_level=risk_level,
            left_regions_with_clinical_context=regions_text
            if "Left" in eye_name
            else "See right eye analysis.",
            right_regions_with_clinical_context=regions_text
            if "Right" in eye_name
            else "See left eye analysis.",
        )

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
            response = await self.client.generate(prompt)

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
        grade_int = _validate_dr_grade(dr_grade)
        grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
            grade_int
        ]

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
        vascular_biomarkers: dict | None,
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

        biomarker_context = ""
        if vascular_biomarkers:
            biomarker_context = (
                f"\nVascular Biomarkers: {json.dumps(vascular_biomarkers)}"
            )

        prompt = f"""You are a medical AI assistant explaining diabetic retinopathy (DR) prediction results.

Explain this prediction in patient-friendly terms addressing these key areas:

1. DIAGNOSIS: The DR grade is {dr_grade} with {confidence:.1%} confidence.

2. FEATURE CONTRIBUTIONS:{shap_context}

3. CLINICAL CONTEXT:{clinical_context}{biomarker_context}

Please provide:
- A clear explanation of what this diagnosis means for the patient
- How the key clinical features influenced the prediction
- Recommended next steps and follow-up actions
- Any warning signs the patient should watch for

Keep the explanation professional but accessible to a non-medical patient."""

        return f"{REPORT_SYSTEM_PROMPT}\n\n{prompt}"

    def _build_gradcam_prompt(
        self,
        left_regions: list[str] | list[dict],
        right_regions: list[str] | list[dict],
    ) -> str:
        left_formatted = []
        for r in left_regions:
            if isinstance(r, dict):
                left_formatted.append(
                    f"{r.get('name', 'unknown')} (intensity: {r.get('intensity', 0):.2f}, "
                    f"area: {r.get('area', 0)} px, saliency: {r.get('saliency_score', 0):.2f})"
                )
            else:
                left_formatted.append(str(r))

        right_formatted = []
        for r in right_regions:
            if isinstance(r, dict):
                right_formatted.append(
                    f"{r.get('name', 'unknown')} (intensity: {r.get('intensity', 0):.2f}, "
                    f"area: {r.get('area', 0)} px, saliency: {r.get('saliency_score', 0):.2f})"
                )
            else:
                right_formatted.append(str(r))

        return f"""Interpret these highlighted regions from GradCAM heatmaps with numerical analysis:

Left Eye: {", ".join(left_formatted) if left_formatted else "No regions detected"}
Right Eye: {", ".join(right_formatted) if right_formatted else "No regions detected"}

Explain what these regions indicate for DR diagnosis, focusing on:
1. Which regions have the highest activation intensity?
2. How does the area of abnormalities compare?
3. What is the clinical significance of the saliency scores?
4. Correlation between intensity values and known DR biomarkers."""

    def _build_severity_prompt(
        self,
        patient_data: dict,
        dr_grade: str,
        risk_factors: list[str],
    ) -> str:
        grade_int = _validate_dr_grade(dr_grade)
        grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
            grade_int
        ]

        patient_info = f"Name: {patient_data.get('name', 'Unknown')}, Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}"
        risk_factors_str = ", ".join(risk_factors) if risk_factors else "None provided"

        return f"""{XAI_SEVERITY_SYSTEM_PROMPT}

PATIENT DATA:
{patient_info}

DIAGNOSIS:
- DR Grade: {grade_int} ({grade_label})
- Risk Factors: {risk_factors_str}

Generate structured severity report as JSON."""

    def _determine_risk_level(self, dr_grade: str | int) -> str:
        # Handle both integer and string inputs
        if isinstance(dr_grade, int) or (
            isinstance(dr_grade, str) and dr_grade.isdigit()
        ):
            return _GRADE_INT_TO_RISK.get(int(dr_grade), "moderate")
        return _GRADE_LABEL_TO_RISK.get(dr_grade, "moderate")

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


_xai_pipeline: XAIPipeline | None = None


def get_xai_pipeline() -> XAIPipeline:
    """FastAPI dependency factory. Creates instance if not overridden."""
    global _xai_pipeline
    if _xai_pipeline is None:
        _xai_pipeline = XAIPipeline()
    return _xai_pipeline
