from __future__ import annotations

import re
import time

from loguru import logger

from app.core.config import settings
from app.vectorstore.chroma_store import ChromaStore
from app.llm.client import get_llm_client
from app.llm.fallback import generate_with_fallback
from app.prompts.templates import (
    GRADCAM_SYSTEM_PROMPT,
    GRADCAM_USER_PROMPT,
    REPORT_SYSTEM_PROMPT,
)
from app.services.prometheus_metrics import (
    XAI_GRADCAM_LATENCY,
    XAI_GRADCAM_REGION_COUNT,
    XAI_GRADCAM_REQUESTS_TOTAL,
    XAI_GRADCAM_STRUCTURE_OK,
    XAI_PREDICTION_LATENCY,
    XAI_PREDICTION_REQUESTS_TOTAL,
    XAI_RAG_AVAILABLE,
)
from app.services.event_client import send_xai_event


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

        client_kwargs: dict[str, str | int] = {
            "model": settings.resolved_model,
            "timeout_seconds": 60,
            "max_tokens": min(settings.max_tokens, 1024),
        }
        if provider == "github":
            client_kwargs["token"] = settings.github_token or ""
            client_kwargs["endpoint"] = settings.github_endpoint
        elif provider == "nvidia":
            client_kwargs["api_key"] = settings.nvidia_api_key or ""
            client_kwargs["base_url"] = settings.nvidia_base_url
        elif provider == "ollama":
            client_kwargs["base_url"] = settings.ollama_base_url
        else:
            client_kwargs["token"] = settings.llm_api_key or ""
            client_kwargs["base_url"] = settings.llm_base_url or ""

        self.client = get_llm_client(provider, **client_kwargs)
        logger.info("XAI Pipeline initialized")

    async def explain_prediction(
        self,
        prediction_id: str,
        dr_grade: str,
        confidence: float,
        gradcam_regions: dict | None = None,
    ) -> dict:
        """Generate natural language explanation of DR prediction using GradCAM regions."""
        logger.info(
            f"explain_prediction_input: pred={prediction_id} grade={dr_grade} conf={confidence} "
            f"gradcam_left={len(gradcam_regions.get('left_eye', [])) if gradcam_regions else None} "
            f"gradcam_right={len(gradcam_regions.get('right_eye', [])) if gradcam_regions else None}"
        )

        await send_xai_event(
            event="xai.prediction",
            stage="prediction",
            status="started",
            progress=0,
            message="Generating prediction explanation...",
            prediction_id=prediction_id,
        )

        start_time = time.time()
        XAI_PREDICTION_REQUESTS_TOTAL.labels(status="started").inc()

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
                logger.info(
                    f"Imaging explanation computed for prediction {prediction_id} using GradCAM regions"
                )

            except Exception as imaging_error:
                logger.warning(
                    f"Imaging explanation calculation failed, continuing without it: {imaging_error}"
                )
                imaging_explanation = None

        await send_xai_event(
            event="xai.prediction",
            stage="prediction",
            status="progress",
            progress=50,
            message="Generating narrative explanation...",
            prediction_id=prediction_id,
        )

        try:
            prompt = self._build_imaging_prompt_with_regions(
                dr_grade,
                confidence,
                gradcam_regions,
            )

            logger.info(f"Starting LLM generation for prediction {prediction_id}")
            result = await generate_with_fallback(self.client, prompt)
            response = result.content
            logger.info(f"LLM generation completed for prediction {prediction_id}")

            XAI_PREDICTION_LATENCY.observe(time.time() - start_time)
            XAI_PREDICTION_REQUESTS_TOTAL.labels(status="completed").inc()

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
            if imaging_explanation:
                result_details["explanation_type"] = "gradcam_regions"
                result_details["top_regions"] = imaging_explanation.to_dict().get("top_regions", [])

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
                "model_used": settings.resolved_model,
                "status": "completed",
                "explanation_type": "gradcam_regions" if imaging_explanation else "basic",
            }
        except Exception as e:
            XAI_PREDICTION_REQUESTS_TOTAL.labels(status="failed").inc()
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
    ) -> str:
        """Build prompt for imaging-based explanation using GradCAM regions.

        Includes per-region model attribution (intensity, saliency) and asks the LLM
        to explain how the model arrived at its prediction, not just describe textbook pathology.
        """
        grade_int = _validate_dr_grade(dr_grade)
        grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
            grade_int
        ]
        risk_level = _GRADE_INT_TO_RISK.get(grade_int, "moderate")

        # Build per-eye region analysis with numeric GradCAM attribution
        left_eye_analysis = ""
        right_eye_analysis = ""
        hotspots_text = ""

        if gradcam_regions:
            left_regions = gradcam_regions.get("left_eye", [])
            right_regions = gradcam_regions.get("right_eye", [])

            if left_regions:
                left_clinical = []
                for region in left_regions:
                    if isinstance(region, dict):
                        name = region.get("name", "unknown")
                        intensity = region.get("intensity", 0.0)
                        saliency = region.get("saliency_score", 0.0)
                    else:
                        name = region
                        intensity = 0.0
                        saliency = 0.0
                    if isinstance(region, dict):
                        region_lesions = region.get("lesions", [])
                    else:
                        region_lesions = []
                    lesion_str = (
                        f"Detected: {', '.join(region_lesions)}."
                        if region_lesions
                        else ""
                    )
                    clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                        name,
                        {
                            "significance": "Retinal region",
                            "high_contribution": "DR-related changes",
                        },
                    )
                    pathology = (
                        lesion_str
                        if region_lesions
                        else self._get_pathology_for_grade(grade_int, name)
                    )
                    left_clinical.append(
                        f"**{name}** (model saliency: {saliency:.3f}, activation intensity: {intensity:.3f}): "
                        f"{clinical_info['significance']}. "
                        f"At grade {grade_int} ({grade_label}), expected pathology: {pathology}. "
                        f"Model detected: {clinical_info.get('high_contribution', 'DR-related changes')}."
                    )
                left_eye_analysis = "\n".join(left_clinical)

            if right_regions:
                right_clinical = []
                for region in right_regions:
                    if isinstance(region, dict):
                        name = region.get("name", "unknown")
                        intensity = region.get("intensity", 0.0)
                        saliency = region.get("saliency_score", 0.0)
                    else:
                        name = region
                        intensity = 0.0
                        saliency = 0.0
                    if isinstance(region, dict):
                        region_lesions = region.get("lesions", [])
                    else:
                        region_lesions = []
                    lesion_str = (
                        f"Detected: {', '.join(region_lesions)}."
                        if region_lesions
                        else ""
                    )
                    clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                        name,
                        {
                            "significance": "Retinal region",
                            "high_contribution": "DR-related changes",
                        },
                    )
                    pathology = (
                        lesion_str
                        if region_lesions
                        else self._get_pathology_for_grade(grade_int, name)
                    )
                    right_clinical.append(
                        f"**{name}** (model saliency: {saliency:.3f}, activation intensity: {intensity:.3f}): "
                        f"{clinical_info['significance']}. "
                        f"At grade {grade_int} ({grade_label}), expected pathology: {pathology}. "
                        f"Model detected: {clinical_info.get('high_contribution', 'DR-related changes')}."
                    )
                right_eye_analysis = "\n".join(right_clinical)

            # Build top-hotspot list across both eyes
            all_with_scores = []
            for r in left_regions:
                if isinstance(r, dict):
                    all_with_scores.append(
                        (
                            "OS",
                            r.get("name", ""),
                            r.get("saliency_score", 0.0),
                            r.get("intensity", 0.0),
                        )
                    )
            for r in right_regions:
                if isinstance(r, dict):
                    all_with_scores.append(
                        (
                            "OD",
                            r.get("name", ""),
                            r.get("saliency_score", 0.0),
                            r.get("intensity", 0.0),
                        )
                    )
            all_with_scores.sort(key=lambda x: x[2], reverse=True)
            if all_with_scores:
                hotspot_lines = [
                    f"  {i + 1}. {name} ({side}) — saliency: {sal:.3f}, intensity: {inten:.3f}"
                    for i, (side, name, sal, inten) in enumerate(all_with_scores[:5])
                ]
                hotspots_text = "\n" + "\n".join(hotspot_lines)

        prompt = f"""You are a retinal specialist explaining how the DR grading model arrived at its prediction.

PATIENT STATUS:
- DR Grade: {grade_int} ({grade_label})
- Model Confidence: {confidence:.1%}
- Risk Level: {risk_level}

GRADCAM HEATMAP ATTRIBUTION — PER-REGION MODEL ACTIVATION:

LEFT EYE (OS) REGIONS (ranked by model saliency):
{left_eye_analysis if left_eye_analysis else "No regions activated."}

RIGHT EYE (OD) REGIONS (ranked by model saliency):
{right_eye_analysis if right_eye_analysis else "No regions activated."}

TOP HOTSPOTS ACROSS BOTH EYES:{hotspots_text}

OUTPUT FORMAT — Use these exact headers to separate per-eye analysis:

### LEFT EYE (OS):
[analysis of each left eye region — do NOT discuss right eye regions here]

### RIGHT EYE (OD):
[analysis of each right eye region — do NOT discuss left eye regions here]

For each eye, explain:
1. Which regions had the HIGHEST model saliency scores and WHY those drove the prediction.
   The model's attention was not uniform — higher saliency = stronger pattern match.
2. What specific image features (microaneurysms, dot-blot hemorrhages, exudates, 
   venous beading, IRMA, neovascularization) did the model likely detect in 
   high-saliency regions that justified the {grade_label} classification?
3. How does the DISTRIBUTION of saliency across regions explain the model's 
   {confidence:.0%} confidence? (e.g., "one region dominated" vs "multiple weak signals")
4. Why did the model predict grade {grade_int} specifically — what features in the 
   activated regions are characteristic of {grade_label} NPDR rather than lower/higher grades?
5. Provide follow-up recommendations based on which specific regions showed activation.

Use precise clinical terminology. Write as a retinal specialist documenting findings."""

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

    def _format_region_for_prompt(self, region: str | dict) -> str:
        """Format a single region (str or dict) with numeric attribution for prompts.

        Handles both plain strings (backward compat) and full dicts with intensity/saliency.
        """
        if isinstance(region, dict):
            name = region.get("name", "unknown")
            intensity = region.get("intensity", 0.0)
            area = region.get("area", 0)
            saliency = region.get("saliency_score", 0.0)
            clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                name,
                {
                    "significance": "Retinal region",
                    "high_contribution": "DR-related changes",
                },
            )
            return (
                f"- {name} (intensity: {intensity:.3f}, saliency: {saliency:.3f}, area: {area}px): "
                f"{clinical_info['significance']}. "
                f"{clinical_info.get('high_contribution', 'Model activation pattern')}."
            )
        else:
            clinical_info = ShapService.REGION_CLINICAL_RELEVANCE.get(
                region,
                {
                    "significance": "Retinal region",
                    "high_contribution": "DR-related changes",
                },
            )
            return (
                f"- {region}: {clinical_info['significance']}. "
                f"{clinical_info.get('high_contribution', 'Model activation pattern')}."
            )

    def _rank_regions_by_saliency(self, regions: list[str | dict]) -> list[dict]:
        """Rank regions by saliency score or intensity, handling str/dict mix."""
        parsed = []
        for r in regions:
            if isinstance(r, dict):
                parsed.append(
                    {
                        "name": r.get("name", "unknown"),
                        "intensity": r.get("intensity", 0.0),
                        "area": r.get("area", 0),
                        "saliency_score": r.get("saliency_score", 0.0),
                    }
                )
            else:
                parsed.append(
                    {
                        "name": r,
                        "intensity": 0.0,
                        "area": 0,
                        "saliency_score": 0.0,
                    }
                )
        parsed.sort(key=lambda x: x["saliency_score"], reverse=True)
        return parsed

    def _retrieve_rag_context(self, query_text: str) -> tuple[str, float]:
        start_time = time.time()
        try:
            store = ChromaStore(
                settings.rag_chroma_persist_directory,
                settings.rag_chroma_collection_name,
                settings.resolved_rag_embedding_model,
            )
            results = store.query(query_text, top_k=2)
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return "", 0.0

        if not results:
            return "", time.time() - start_time

        snippets = []
        for doc, _score in results:
            text = getattr(doc, "page_content", str(doc)).strip()
            metadata = getattr(doc, "metadata", {}) or {}
            if text:
                snippets.append(
                    f"[source: {metadata.get('artifact_id', 'unknown')}] {text}"
                )

        context = "\n".join(snippets)
        return context, time.time() - start_time

    async def explain_gradcam(
        self,
        prediction_id: str,
        left_eye_regions: list[str | dict],
        right_eye_regions: list[str | dict],
        dr_grade: str | int | None = None,
        confidence: float | None = None,
    ) -> dict:
        """Interpret highlighted regions in GradCAM heatmaps with model-attribution explanation.

        Uses a single LLM call with both eyes' regions and their numeric GradCAM values
        (intensity, saliency) to produce a model-explanatory clinical narrative.

        Args:
            prediction_id: Unique identifier for the prediction.
            left_eye_regions: Region names or dicts with name/intensity/area/saliency_score.
            right_eye_regions: Region names or dicts with name/intensity/area/saliency_score.
            dr_grade: Optional DR grade (0-4) for clinical context.
            confidence: Optional model confidence (0-1) for clinical context.

        Returns:
            dict with left_eye_explanation and right_eye_explanation.
        """
        logger.info(
            f"gradcam_input_data: pred={prediction_id} dr={dr_grade} conf={confidence} "
            f"left_regions({len(left_eye_regions)})={left_eye_regions} "
            f"right_regions({len(right_eye_regions)})={right_eye_regions}"
        )

        await send_xai_event(
            event="xai.gradcam",
            stage="gradcam",
            status="started",
            progress=0,
            message="Interpreting GradCAM regions with model attribution...",
            prediction_id=prediction_id,
        )

        start_time = time.time()
        XAI_GRADCAM_REQUESTS_TOTAL.labels(status="started").inc()
        XAI_GRADCAM_REGION_COUNT.labels(eye="left").set(len(left_eye_regions))
        XAI_GRADCAM_REGION_COUNT.labels(eye="right").set(len(right_eye_regions))

        try:
            grade_int = _validate_dr_grade(dr_grade) if dr_grade is not None else 2
            grade_label = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"][
                grade_int
            ]
            conf = confidence if confidence is not None else 0.75
            risk_level = _GRADE_INT_TO_RISK.get(grade_int, "moderate")

            # Rank regions by actual GradCAM saliency
            left_ranked = self._rank_regions_by_saliency(left_eye_regions)
            right_ranked = self._rank_regions_by_saliency(right_eye_regions)

            # Build per-eye region analyses with numeric attribution
            left_sections = []
            for i, r in enumerate(left_ranked):
                line = self._format_region_for_prompt(r)
                pct = r["saliency_score"] * 100 if r["saliency_score"] > 0 else None
                if pct and len(left_ranked) > 1:
                    line += f" [Rank {i + 1}/{len(left_ranked)}]"
                left_sections.append(line)

            right_sections = []
            for i, r in enumerate(right_ranked):
                line = self._format_region_for_prompt(r)
                pct = r["saliency_score"] * 100 if r["saliency_score"] > 0 else None
                if pct and len(right_ranked) > 1:
                    line += f" [Rank {i + 1}/{len(right_ranked)}]"
                right_sections.append(line)

            left_text = (
                "\n".join(left_sections) if left_sections else "No regions highlighted."
            )
            right_text = (
                "\n".join(right_sections)
                if right_sections
                else "No regions highlighted."
            )

            # Top hotspots across both eyes
            all_ranked = sorted(
                left_ranked + right_ranked,
                key=lambda x: x["saliency_score"],
                reverse=True,
            )
            hotspots = []
            for i, r in enumerate(all_ranked[:5]):
                side = "OS" if r in left_ranked else "OD"
                hotspots.append(
                    f"  {i + 1}. {r['name']} ({side}) — saliency: {r['saliency_score']:.3f}, intensity: {r['intensity']:.3f}"
                )
            hotspots_text = "\n".join(hotspots) if hotspots else "  No hotspot data."

            # Build RAG search query from top-3 highest-saliency regions
            top_region_names = [r["name"] for r in all_ranked[:3] if r.get("name")]
            query_text = (
                f"{' '.join(top_region_names)} {grade_label} GradCAM heatmap activation"
            )
            retrieved_context, retrieval_time = self._retrieve_rag_context(query_text)
            if retrieved_context:
                logger.info(
                    f"RAG context retrieved in {retrieval_time:.2f}s ({len(retrieved_context)} chars)"
                )
            else:
                logger.debug("No RAG context retrieved for GradCAM analysis")

            rag_section = ""
            if retrieved_context:
                rag_section = f"""
RETRIEVED CLINICAL CONTEXT:
{retrieved_context}

Use this literature as clinical reference. Note if the model's findings align with or diverge from it.
"""

            item4_note = ""
            if not retrieved_context:
                item4_note = " (or note that RAG context was unavailable)"

            prompt = f"""Analyze the GradCAM heatmap activations for diabetic retinopathy diagnosis.

PATIENT DR STATUS:
- DR Grade: {grade_int} ({grade_label})
- Model Confidence: {conf:.1%}
- Risk Level: {risk_level}

LEFT EYE (OS) REGIONS RANKED BY SALIENCY:
{left_text}

RIGHT EYE (OD) REGIONS RANKED BY SALIENCY:
{right_text}

TOP HOTSPOTS ACROSS BOTH EYES:
{hotspots_text}
{rag_section}
OUTPUT FORMAT — Use these exact headers to separate per-eye analysis:

### LEFT EYE (OS):
[detailed analysis of each left eye region — do NOT discuss right eye regions here]
For EACH highlighted region, cover:
1. What is its saliency score and how does its rank among all regions indicate 
   the model's attention allocation? Higher saliency = stronger evidence.
2. Given the region's activation intensity and the model's overall confidence, 
   what specific image features (microaneurysms, hemorrhages, exudates, 
   neovascularization, venous beading, IRMA) did the model likely detect here?
3. How do these per-region activations explain WHY the model predicted 
   grade {grade_int} instead of a lower or higher grade?
4. Why is the model {conf:.0%} confident rather than higher or lower — which 
   regions had the strongest feature matches and which were borderline?{item4_note}

### RIGHT EYE (OD):
[detailed analysis of each right eye region — do NOT discuss left eye regions here]
For EACH highlighted region, cover:
1. What is its saliency score and how does its rank among all regions indicate 
   the model's attention allocation? Higher saliency = stronger evidence.
2. Given the region's activation intensity and the model's overall confidence, 
   what specific image features (microaneurysms, hemorrhages, exudates, 
   neovascularization, venous beading, IRMA) did the model likely detect here?
3. How do these per-region activations explain WHY the model predicted 
   grade {grade_int} instead of a lower or higher grade?
4. Why is the model {conf:.0%} confident rather than higher or lower — which 
   regions had the strongest feature matches and which were borderline?{item4_note}

Use precise clinical terminology. Write as a retinal specialist documenting 
findings. Output complete clinical narrative only."""

            combined = f"{GRADCAM_SYSTEM_PROMPT}\n\n{prompt}"
            result = await generate_with_fallback(self.client, combined)
            response = result.content

            XAI_GRADCAM_LATENCY.observe(time.time() - start_time)
            XAI_GRADCAM_REQUESTS_TOTAL.labels(status="completed").inc()

            has_region_names = bool(
                re.search(
                    r"(macula|fovea|periphery|arcade|disk)\b", response, re.IGNORECASE
                )
            )
            has_saliency = bool(re.search(r"\d+\.\d{2,}", response))
            XAI_GRADCAM_STRUCTURE_OK.labels(
                result="pass" if has_region_names and has_saliency else "fail"
            ).inc()

            # Parse per-eye sections from the response
            left_match = re.search(
                r"(?:###\s*|\*\*)\s*LEFT EYE\s*\(?OS\)?\s*(?:\*\*)?\s*:\s*\n+(.*?)(?=\n*(?:###\s*|\*\*)\s*RIGHT EYE\s*\(?OD\)?\s*(?:\*\*)?\s*:\s*|\Z)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            right_match = re.search(
                r"(?:###\s*|\*\*)\s*RIGHT EYE\s*\(?OD\)?\s*(?:\*\*)?\s*:\s*\n+(.*?)(?=\Z)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            left_explanation = left_match.group(1).strip() if left_match else ""
            right_explanation = right_match.group(1).strip() if right_match else ""
            if not left_explanation or not right_explanation:
                logger.warning(
                    "gradcam_parse_fallback",
                    has_left=bool(left_explanation),
                    has_right=bool(right_explanation),
                    response_length=len(response),
                )
            if not left_explanation and not right_explanation:
                left_explanation = response
                right_explanation = response

            highlighted_region_names = {
                "left_eye": [
                    r["name"] if isinstance(r, dict) else r for r in left_eye_regions
                ],
                "right_eye": [
                    r["name"] if isinstance(r, dict) else r for r in right_eye_regions
                ],
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
                    "left_eye": left_explanation,
                    "right_eye": right_explanation,
                    "highlighted_regions": highlighted_region_names,
                    "dr_grade": grade_label,
                    "confidence": conf,
                },
            )

            logger.info(
                f"gradcam_output_data: pred={prediction_id} grade={grade_label} conf={conf} "
                f"left_len={len(left_explanation)} right_len={len(right_explanation)}"
            )

            return {
                "left_eye_explanation": left_explanation,
                "right_eye_explanation": right_explanation,
                "highlighted_regions": highlighted_region_names,
                "dr_grade": grade_label,
                "confidence": conf,
                "model_used": settings.resolved_model,
            }
        except Exception as e:
            XAI_GRADCAM_REQUESTS_TOTAL.labels(status="failed").inc()
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
            result = await generate_with_fallback(self.client, prompt)
            response = result.content

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
                "model_used": settings.resolved_model,
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
3. What is the clinical significance of the saliency scores?"""

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
