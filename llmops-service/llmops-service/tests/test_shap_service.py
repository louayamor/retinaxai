from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.services.shap_service import (
    FeatureContribution,
    ImagingExplanation,
    ShapExplainabilityError,
    ShapExplanation,
    ShapService,
)


class DummyModel:
    def __init__(self, n_features_in_: int = 17):
        self.n_features_in_ = n_features_in_

    def predict_proba(self, x):
        return np.array([[0.1, 0.2, 0.3, 0.2, 0.2]])


class TestShapService:
    @pytest.fixture
    def service(self) -> ShapService:
        return ShapService()

    def test_feature_encoding_returns_correct_length(self, service: ShapService):
        features = {
            "thickness_center_fovea": 250.0,
            "thickness_average_thickness": 280.0,
            "thickness_total_volume_mm3": 8.5,
            "thickness_inner_superior": 300.0,
            "thickness_inner_nasal": 290.0,
            "thickness_inner_inferior": 295.0,
            "thickness_inner_temporal": 305.0,
            "thickness_outer_superior": 310.0,
            "thickness_outer_nasal": 315.0,
            "thickness_outer_inferior": 320.0,
            "thickness_outer_temporal": 325.0,
            "patient_age": 65,
            "patient_gender": "M",
            "meta_eye": "OD",
            "clinical_edema": "True",
            "clinical_erm_status": "present",
            "meta_image_quality": 0.9,
        }
        encoded = service._encode_features(features)
        assert len(encoded) == 17
        assert encoded[12] == 1.0  # male
        assert encoded[13] == 1.0  # OD
        assert encoded[14] == 1.0  # edema True
        assert encoded[15] == 0.0  # erm present
        assert encoded[16] == 0.9  # quality

    def test_feature_encoding_handles_missing_keys(self, service: ShapService):
        encoded = service._encode_features({})
        assert len(encoded) == 17
        assert all(v == 0.0 for v in encoded)

    def test_feature_encoding_handles_lists(self, service: ShapService):
        encoded = service._encode_features({"patient_age": [55]})
        assert encoded[11] == 55.0

    def test_feature_encoding_handles_none(self, service: ShapService):
        encoded = service._encode_features({"patient_age": None})
        assert encoded[11] == 0.0

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

    def test_explain_prediction_with_mock_model(self, service: ShapService):
        model = DummyModel(n_features_in_=17)
        feature_names = service._get_feature_names()
        features = {name: float(i) for i, name in enumerate(feature_names)}

        async def fake_load():
            return model

        from unittest.mock import patch

        with patch.object(service, "_load_clinical_model", new=fake_load):
            result = asyncio.run(
                service.explain_prediction(features, pipeline="clinical")
            )

        assert isinstance(result, ShapExplanation)
        assert result.pipeline == "clinical"
        assert len(result.contributions) == 17
        d = result.to_dict()
        assert "top_positive" in d
        assert "top_negative" in d

    def test_explain_prediction_raises_on_feature_count_mismatch(
        self, service: ShapService
    ):
        model = DummyModel(n_features_in_=10)
        features = {f"feat_{i}": float(i) for i in range(17)}

        async def fake_load():
            return model

        from unittest.mock import patch

        with patch.object(service, "_load_clinical_model", new=fake_load):
            with pytest.raises(ShapExplainabilityError) as exc_info:
                asyncio.run(service.explain_prediction(features, pipeline="clinical"))

            assert "Feature count mismatch: expected 10, got 17" in str(exc_info.value)

    def test_explain_prediction_raises_when_n_features_missing(
        self, service: ShapService
    ):
        model = object()  # No n_features_in_ attribute
        features = {f"feat_{i}": float(i) for i in range(17)}

        async def fake_load():
            return model

        from unittest.mock import patch

        with patch.object(service, "_load_clinical_model", new=fake_load):
            with pytest.raises(ShapExplainabilityError) as exc_info:
                asyncio.run(service.explain_prediction(features, pipeline="clinical"))

            assert "Cannot determine expected feature count" in str(exc_info.value)

    def test_shap_explanation_to_dict(self):
        contributions = [
            FeatureContribution("feat_a", 0.5, 0.0, 1.0),
            FeatureContribution("feat_b", -0.3, 0.0, 2.0),
        ]
        explanation = ShapExplanation("xgboost", 0.5, contributions, "clinical")
        d = explanation.to_dict()
        assert d["model_type"] == "xgboost"
        assert d["expected_value"] == 0.5
        assert len(d["features"]) == 2
        assert d["top_positive"][0]["name"] == "feat_a"
        assert d["top_negative"][0]["name"] == "feat_b"

    def test_global_importance_returns_empty_when_no_shap(
        self, service: ShapService, tmp_path
    ):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("patient_age,patient_gender\n55,M\n60,F\n")
        result = asyncio.run(
            service.compute_global_importance(csv_path, pipeline="clinical")
        )
        assert result == {}
