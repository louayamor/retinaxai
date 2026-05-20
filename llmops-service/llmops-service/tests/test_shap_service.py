from __future__ import annotations

import pytest

from app.services.shap_service import (
    ImagingExplanation,
    ShapService,
)


class TestShapService:
    @pytest.fixture
    def service(self) -> ShapService:
        return ShapService()

    def test_explain_imaging_prediction(self, service: ShapService):
        regions = {
            "left_eye": ["fovea_centralis", "macula_center"],
            "right_eye": ["optic_disk_nasal"],
        }
        explanation = service.explain_imaging_prediction(
            regions, prediction_grade=3, confidence=0.85
        )
        assert isinstance(explanation, ImagingExplanation)
        assert explanation.prediction_grade == 3
        assert explanation.confidence == 0.85
        assert len(explanation.regions) == 3
        d = explanation.to_dict()
        assert "top_regions" in d
        assert len(d["top_regions"]) <= 5

    def test_explain_imaging_prediction_empty_regions_fallback(
        self, service: ShapService
    ):
        regions = {"left_eye": [], "right_eye": []}
        explanation = service.explain_imaging_prediction(
            regions, prediction_grade=1, confidence=0.7
        )
        assert len(explanation.regions) >= 1

    def test_explain_imaging_prediction_with_dict_regions(self, service: ShapService):
        regions = {
            "left_eye": [
                {"name": "fovea_centralis", "intensity": 0.9, "saliency_score": 0.85}
            ],
            "right_eye": [],
        }
        explanation = service.explain_imaging_prediction(
            regions, prediction_grade=3, confidence=0.85
        )
        assert isinstance(explanation, ImagingExplanation)
        assert len(explanation.regions) == 1
        assert explanation.regions[0].contribution == 0.85

    def test_imaging_explanation_to_dict(self):
        from app.services.shap_service import ImagingFeatureContribution

        regions = [
            ImagingFeatureContribution(
                region_name="fovea_centralis",
                contribution=0.8,
                anatomical_significance="Central vision focal point",
                clinical_relevance="Microaneurysms indicate severe DR progression",
            ),
        ]
        explanation = ImagingExplanation(
            "gradcam_efficientnet", 3, 0.85, regions, "imaging"
        )
        d = explanation.to_dict()
        assert d["model_type"] == "gradcam_efficientnet"
        assert d["prediction_grade"] == 3
        assert d["confidence"] == 0.85
        assert len(d["top_regions"]) == 1
