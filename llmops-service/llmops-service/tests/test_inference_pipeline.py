from __future__ import annotations

import asyncio
import json

import pytest

from app.core.config import LLMProvider
from app.llm.client import MockLLMClient
from app.pipeline import inference_pipeline as ip
from app.pipeline import xai_pipeline as xp
from app.pipeline.inference_pipeline import InferencePipeline, get_inference_pipeline
from app.pipeline.xai_pipeline import InvalidGradeError, XAIPipeline, _validate_dr_grade
from app.prompts.templates import REPORT_USER_PROMPT, _safe_format


def test_inference_pipeline_uses_mock_client(monkeypatch):
    monkeypatch.setattr(ip.settings, "llm_provider", LLMProvider.MOCK)

    pipeline = InferencePipeline()
    assert isinstance(pipeline.client, MockLLMClient)


def test_generate_report_returns_required_keys(monkeypatch):
    monkeypatch.setattr(ip.settings, "llm_provider", LLMProvider.MOCK)

    pipeline = InferencePipeline()
    result = asyncio.run(
        pipeline.generate_report(
            {
                "patient": {"id": "P001", "age": 60},
                "prediction": {"grade": 1},
                "cleaned_summary": "No significant findings.",
                "raw_ocr_text": "",
                "report_type": "report",
                "language": "en",
                "tone": "clinical",
            }
        )
    )

    assert "content" in result
    assert "summary" in result
    assert "model_used" in result


def test_generate_report_with_json_response(monkeypatch):
    monkeypatch.setattr(ip.settings, "llm_provider", LLMProvider.MOCK)

    pipeline = InferencePipeline()

    async def fake_generate(
        prompt: str, system_prompt: str | None = None, **kwargs
    ) -> str:
        return json.dumps({"content": "Full report text.", "summary": "Short summary."})

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)

    result = asyncio.run(pipeline.generate_report({"report_type": "report"}))
    assert result["content"] == "Full report text."
    assert result["summary"] == "Short summary."


def test_generate_report_with_plain_text_response(monkeypatch):
    monkeypatch.setattr(ip.settings, "llm_provider", LLMProvider.MOCK)

    pipeline = InferencePipeline()

    plain_text = "This is a plain text report without JSON."

    async def fake_generate(
        prompt: str, system_prompt: str | None = None, **kwargs
    ) -> str:
        return plain_text

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)

    result = asyncio.run(pipeline.generate_report({"report_type": "report"}))
    assert result["content"] == plain_text
    assert result["summary"] == plain_text[:400]


def test_generate_report_includes_retrieved_context(monkeypatch):
    monkeypatch.setattr(ip.settings, "llm_provider", LLMProvider.MOCK)

    pipeline = InferencePipeline()

    class DummyStore:
        def query(self, text: str, top_k: int = 4) -> list:
            class Doc:
                page_content = "retrieved chunk"
                metadata = {"artifact_id": "clinical_metrics"}

            return [[Doc()]]

    pipeline.store = DummyStore()  # type: ignore[assignment]

    captured: dict[str, str] = {}

    async def fake_generate(
        prompt: str, system_prompt: str | None = None, **kwargs
    ) -> str:
        captured["prompt"] = prompt
        return '{"content": "ok", "summary": "ok"}'

    monkeypatch.setattr(pipeline.client, "generate", fake_generate)

    asyncio.run(
        pipeline.generate_report(
            {"cleaned_summary": "summary text", "raw_ocr_text": "ocr text"}
        )
    )

    assert "REFERENCE CONTEXT:" in captured["prompt"]
    assert "retrieved chunk" in captured["prompt"]


def test_safe_format_escapes_braces():
    template = "Patient: {name}, Notes: {notes}"
    result = _safe_format(
        template,
        name="John",
        notes="has {special} condition",
    )
    assert result == "Patient: John, Notes: has {special} condition"


def test_safe_format_with_json_value():
    template = "Data: {data}"
    result = _safe_format(template, data='{"key": "value with {brace}"}')
    assert result == 'Data: {"key": "value with {brace}"}'


def test_safe_format_prevents_keyerror_injection():
    template = "Report: {report}"
    result = _safe_format(template, report="{missing_key}")
    assert result == "Report: {missing_key}"


class TestXAIPipelineValidation:
    def test_validate_dr_grade_with_valid_int(self):
        assert _validate_dr_grade(0) == 0
        assert _validate_dr_grade(4) == 4

    def test_validate_dr_grade_with_valid_string(self):
        assert _validate_dr_grade("2") == 2
        assert _validate_dr_grade("0") == 0

    def test_validate_dr_grade_with_invalid_string_raises(self):
        with pytest.raises(InvalidGradeError):
            _validate_dr_grade("moderate")

    def test_validate_dr_grade_with_negative_int_raises(self):
        with pytest.raises(InvalidGradeError):
            _validate_dr_grade(-1)

    def test_validate_dr_grade_with_high_int_raises(self):
        with pytest.raises(InvalidGradeError):
            _validate_dr_grade(5)

    def test_validate_dr_grade_with_float_raises(self):
        with pytest.raises(InvalidGradeError):
            _validate_dr_grade(2.5)

    def test_xai_pipeline_builds_prediction_prompt(self):
        pipeline = XAIPipeline()
        prompt = pipeline._build_prediction_prompt("2", 0.85, {"age": 60})
        assert "DR Grade: 2 (Moderate)" in prompt
        assert "Confidence: 0.85" in prompt

    def test_xai_pipeline_builds_severity_prompt(self):
        pipeline = XAIPipeline()
        prompt = pipeline._build_severity_prompt(
            {"name": "Jane", "age": 55, "gender": "F"},
            "3",
            ["hypertension"],
        )
        assert "DR Grade: 3 (Severe)" in prompt
        assert "hypertension" in prompt

    def test_xai_pipeline_invalid_grade_in_prompt_raises(self):
        pipeline = XAIPipeline()
        with pytest.raises(InvalidGradeError):
            pipeline._build_prediction_prompt("invalid", 0.5, None)


def test_user_prompt_template_rendering_with_safe_format():
    prompt = _safe_format(
        REPORT_USER_PROMPT,
        patient='{"name":"Test"}',
        prediction='{"grade":2}',
        cleaned_summary="summary",
        raw_ocr_text="ocr",
        report_type="report",
        language="en",
        tone="clinical",
        retrieved_context="context",
    )
    assert "PATIENT INFORMATION:" in prompt
    assert '{"name":"Test"}' in prompt
    assert "REFERENCE CONTEXT:" in prompt


def test_get_inference_pipeline_dependency_creates_instance():
    """Test that dependency factory creates instance (not raises)."""
    from app.pipeline.inference_pipeline import InferencePipeline

    pipeline = get_inference_pipeline()
    assert isinstance(pipeline, InferencePipeline)


class TestGradCAMPerEyeAnalysis:
    """Tests for per-eye GradCAM explanation improvements."""

    def test_get_pathology_for_grade_mild_central(self):
        pipeline = XAIPipeline()
        pathology = pipeline._get_pathology_for_grade(1, "macula_center")
        assert "microaneurysms" in pathology.lower()

    def test_get_pathology_for_grade_moderate_central(self):
        pipeline = XAIPipeline()
        pathology = pipeline._get_pathology_for_grade(2, "fovea_centralis")
        assert "microaneurysms" in pathology.lower()
        assert "hemorrhages" in pathology.lower()

    def test_get_pathology_for_grade_severe_vascular(self):
        pipeline = XAIPipeline()
        pathology = pipeline._get_pathology_for_grade(3, "superior_temporal_arcade")
        assert "venous beading" in pathology.lower() or "IRMA" in pathology

    def test_get_pathology_for_grade_proliferative_peripheral(self):
        pipeline = XAIPipeline()
        pathology = pipeline._get_pathology_for_grade(4, "inferior_periphery")
        assert "neovascularization" in pathology.lower()

    def test_get_pathology_for_grade_zero(self):
        pipeline = XAIPipeline()
        pathology = pipeline._get_pathology_for_grade(0, "macula_center")
        assert "No DR" in pathology or "expected" in pathology.lower()

    def test_format_region_for_prompt_with_string(self):
        pipeline = XAIPipeline()
        result = pipeline._format_region_for_prompt("macula_center")
        assert "macula_center" in result
        assert "significance" in result.lower() or "vision" in result.lower()

    def test_format_region_for_prompt_with_dict(self):
        pipeline = XAIPipeline()
        result = pipeline._format_region_for_prompt(
            {
                "name": "fovea_centralis",
                "intensity": 0.85,
                "area": 450,
                "saliency_score": 0.72,
            }
        )
        assert "fovea_centralis" in result
        assert "0.850" in result or "0.85" in result
        assert "0.720" in result or "0.72" in result

    def test_rank_regions_by_saliency_sorts_descending(self):
        pipeline = XAIPipeline()
        regions = [
            {"name": "low", "intensity": 0.3, "area": 100, "saliency_score": 0.2},
            {"name": "high", "intensity": 0.9, "area": 200, "saliency_score": 0.8},
            {"name": "mid", "intensity": 0.6, "area": 150, "saliency_score": 0.5},
        ]
        ranked = pipeline._rank_regions_by_saliency(regions)
        assert [r["name"] for r in ranked] == ["high", "mid", "low"]

    def test_rank_regions_with_string_fallback(self):
        pipeline = XAIPipeline()
        ranked = pipeline._rank_regions_by_saliency(
            ["macula_center", "inferior_periphery"]
        )
        assert len(ranked) == 2
        assert all(r["saliency_score"] == 0.0 for r in ranked)

    def _patch_xai_settings(self, monkeypatch):
        monkeypatch.setattr(xp.settings, "llm_provider", "mock")
        monkeypatch.setattr(xp.settings, "llm_model", "mock-model")
        monkeypatch.setattr(xp.settings, "timeout_seconds", 30)
        monkeypatch.setattr(xp.settings, "max_tokens", 1024)
        monkeypatch.setattr(xp.settings, "github_token", "")
        monkeypatch.setattr(xp.settings, "github_endpoint", "")
        monkeypatch.setattr(xp.settings, "llm_api_key", "")
        monkeypatch.setattr(xp.settings, "llm_base_url", "")
        monkeypatch.setattr(xp.settings, "ollama_base_url", "")

    def test_explain_gradcam_returns_required_keys(self, monkeypatch):
        self._patch_xai_settings(monkeypatch)

        pipeline = XAIPipeline()
        result = asyncio.run(
            pipeline.explain_gradcam(
                prediction_id="test-id",
                left_eye_regions=[
                    {
                        "name": "macula_center",
                        "intensity": 0.85,
                        "area": 450,
                        "saliency_score": 0.72,
                    },
                ],
                right_eye_regions=[
                    {
                        "name": "inferior_periphery",
                        "intensity": 0.55,
                        "area": 290,
                        "saliency_score": 0.34,
                    },
                ],
                dr_grade=2,
                confidence=0.85,
            )
        )
        assert "left_eye_explanation" in result
        assert "right_eye_explanation" in result
        assert "highlighted_regions" in result
        assert "dr_grade" in result
        assert "confidence" in result
        assert "model_used" in result
        assert result["highlighted_regions"]["left_eye"] == ["macula_center"]

    def test_explain_gradcam_handles_string_regions_fallback(self, monkeypatch):
        self._patch_xai_settings(monkeypatch)

        pipeline = XAIPipeline()
        result = asyncio.run(
            pipeline.explain_gradcam(
                prediction_id="test-id",
                left_eye_regions=["macula_center"],
                right_eye_regions=["inferior_periphery"],
                dr_grade=2,
                confidence=0.85,
            )
        )
        assert result["left_eye_explanation"]
        assert result["right_eye_explanation"]
        assert result["highlighted_regions"]["left_eye"] == ["macula_center"]
