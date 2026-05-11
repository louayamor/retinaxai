from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import numpy as np
from loguru import logger

from app.core.config import settings

_IO_LIMITER = anyio.CapacityLimiter(4)


class ShapExplainabilityError(Exception):
    pass


class FeatureContribution:
    def __init__(
        self,
        feature_name: str,
        contribution: float,
        base_value: float,
        value: float,
    ):
        self.feature_name = feature_name
        self.contribution = contribution
        self.base_value = base_value
        self.value = value


class ImagingFeatureContribution:
    def __init__(
        self,
        region_name: str,
        contribution: float,
        anatomical_significance: str,
        clinical_relevance: str,
    ):
        self.region_name = region_name
        self.contribution = contribution
        self.anatomical_significance = anatomical_significance
        self.clinical_relevance = clinical_relevance


class ShapExplanation:
    def __init__(
        self,
        model_type: str,
        expected_value: float,
        contributions: list[FeatureContribution],
        pipeline: str,
    ):
        self.model_type = model_type
        self.expected_value = expected_value
        self.contributions = contributions
        self.pipeline = pipeline

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "expected_value": self.expected_value,
            "pipeline": self.pipeline,
            "features": [
                {
                    "name": c.feature_name,
                    "contribution": c.contribution,
                    "base_value": c.base_value,
                    "value": c.value,
                }
                for c in self.contributions
            ],
            "top_positive": [
                {
                    "name": c.feature_name,
                    "contribution": c.contribution,
                }
                for c in sorted(
                    self.contributions,
                    key=lambda x: x.contribution,
                    reverse=True,
                )[:5]
            ],
            "top_negative": [
                {
                    "name": c.feature_name,
                    "contribution": c.contribution,
                }
                for c in sorted(
                    self.contributions,
                    key=lambda x: x.contribution,
                )[:5]
            ],
        }


class ImagingExplanation:
    def __init__(
        self,
        model_type: str,
        prediction_grade: int,
        confidence: float,
        regions: list[ImagingFeatureContribution],
        pipeline: str = "imaging",
    ):
        self.model_type = model_type
        self.prediction_grade = prediction_grade
        self.confidence = confidence
        self.regions = regions
        self.pipeline = pipeline

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "prediction_grade": self.prediction_grade,
            "confidence": self.confidence,
            "pipeline": self.pipeline,
            "regions": [
                {
                    "name": r.region_name,
                    "contribution": r.contribution,
                    "anatomical_significance": r.anatomical_significance,
                    "clinical_relevance": r.clinical_relevance,
                }
                for r in self.regions
            ],
            "top_regions": [
                {
                    "name": r.region_name,
                    "contribution": r.contribution,
                    "clinical_relevance": r.clinical_relevance,
                }
                for r in sorted(
                    self.regions,
                    key=lambda x: x.contribution,
                    reverse=True,
                )[:5]
            ],
        }


class ShapService:
    REGION_CLINICAL_RELEVANCE = {
        "fovea_centralis": {
            "significance": "Central vision focal point",
            "high_contribution": "Microaneurysms or exudates at fovea indicate severe DR progression",
            "moderate_contribution": "Changes near fovea may affect central acuity",
        },
        "macula_center": {
            "significance": "Central retinal area for detailed vision",
            "high_contribution": "Diabetic macular edema often localizes here",
            "moderate_contribution": "Macular involvement affects reading and fine detail",
        },
        "superior_macula": {
            "significance": "Upper macular region",
            "high_contribution": "Edema or hemorrhages affecting superior arcade drainage",
            "moderate_contribution": "May impact superior visual field",
        },
        "inferior_macula": {
            "significance": "Lower macular region",
            "high_contribution": "Fluid accumulation common in diabetic macular edema",
            "moderate_contribution": "Affects inferior visual field perception",
        },
        "perifovea": {
            "significance": "Peripheral macular zone",
            "high_contribution": "Peripheral retinal changes may indicate DR progression",
            "moderate_contribution": "Peripheral pathology suggests non-proliferative changes",
        },
        "optic_disk_nasal": {
            "significance": "Optic nerve head region",
            "high_contribution": "Disk swelling or neovascularization indicates advanced DR",
            "moderate_contribution": "Peripapillary changes may affect peripheral vision",
        },
        "nasal_mid_periphery": {
            "significance": "Nasal retinal region",
            "high_contribution": "Cotton wool spots or hemorrhages indicate ischemia",
            "moderate_contribution": "Nasal retinopathy suggests systemic progression",
        },
        "nasal_periphery": {
            "significance": "Outer nasal retina",
            "high_contribution": "Peripheral neovascularization in proliferative DR",
            "moderate_contribution": "Peripheral changes may precede central involvement",
        },
        "superior_nasal_periphery": {
            "significance": "Upper nasal outer retina",
            "high_contribution": "Peripheral neovascularization common in severe DR",
            "moderate_contribution": "May indicate progression from mid-periphery",
        },
        "inferior_nasal_periphery": {
            "significance": "Lower nasal outer retina",
            "high_contribution": "Peripheral pathology in proliferative DR",
            "moderate_contribution": "Inferior peripheral changes require monitoring",
        },
        "temporal_arcade": {
            "significance": "Major temporal vascular arcade",
            "high_contribution": "Arcade hemorrhages indicate venous stasis",
            "moderate_contribution": "Vascular changes suggest hypertensive component",
        },
        "superior_temporal_arcade": {
            "significance": "Upper temporal vasculature",
            "high_contribution": "Venous beading and IRMA indicate severe NPDR",
            "moderate_contribution": "Arcade changes may affect superior field",
        },
        "inferior_temporal_arcade": {
            "significance": "Lower temporal vasculature",
            "high_contribution": "Hemorrhages and microaneurysms along lower arcade",
            "moderate_contribution": "Inferior arcade pathology affects lower visual field",
        },
        "superior_temporal_periphery": {
            "significance": "Upper outer temporal retina",
            "high_contribution": "Peripheral neovascularization in PDR",
            "moderate_contribution": "May require scatter laser treatment",
        },
        "inferior_temporal_periphery": {
            "significance": "Lower outer temporal retina",
            "high_contribution": "Peripheral pathology common in proliferative DR",
            "moderate_contribution": "Requires peripheral retinal examination",
        },
        "temporal_periphery": {
            "significance": "Outer temporal retina",
            "high_contribution": "Peripheral neovascularization requires laser",
            "moderate_contribution": "Temporal peripheral changes indicate progression",
        },
        "superior_arcade": {
            "significance": "Major superior vascular arcade",
            "high_contribution": "Arcade hemorrhages and venous changes",
            "moderate_contribution": "Superior vascular changes affect upper field",
        },
        "inferior_arcade": {
            "significance": "Major inferior vascular arcade",
            "high_contribution": "Venous beading and hemorrhages",
            "moderate_contribution": "Inferior arcade changes affect lower field",
        },
        "superior_periphery": {
            "significance": "Upper peripheral retina",
            "high_contribution": "Peripheral neovascularization zone",
            "moderate_contribution": "May require panretinal photocoagulation",
        },
        "inferior_periphery": {
            "significance": "Lower peripheral retina",
            "high_contribution": "Peripheral pathology in PDR",
            "moderate_contribution": "Inferior peripheral changes need monitoring",
        },
        "mid_periphery": {
            "significance": "Mid-peripheral retina",
            "high_contribution": "Peripheral retinopathy changes",
            "moderate_contribution": "Mid-peripheral involvement indicates progression",
        },
        "posterior_pole": {
            "significance": "Central posterior retina",
            "high_contribution": "Posterior pole involvement affects central vision",
            "moderate_contribution": "May indicate diabetic macular edema",
        },
    }

    def __init__(self):
        self.artifacts_root = settings.artifacts_root
        self._shap_values_cache: dict[str, list[dict]] = {}
        self._global_importance: dict[str, dict[str, float]] = {}

    def _get_region_contribution(
        self, region: str, grade: int, confidence: float
    ) -> float:
        """Calculate contribution score for an anatomical region based on DR grade."""
        base_score = confidence * 0.5

        grade_multiplier = {
            0: 0.2,
            1: 0.4,
            2: 0.6,
            3: 0.85,
            4: 1.0,
        }.get(grade, 0.5)

        central_regions = [
            "fovea_centralis",
            "macula_center",
            "perifovea",
            "superior_macula",
            "inferior_macula",
            "posterior_pole",
        ]
        peripheral_regions = [
            "nasal_periphery",
            "temporal_periphery",
            "superior_periphery",
            "inferior_periphery",
            "superior_temporal_periphery",
            "inferior_temporal_periphery",
            "superior_nasal_periphery",
            "inferior_nasal_periphery",
        ]
        vascular_regions = [
            "temporal_arcade",
            "superior_temporal_arcade",
            "inferior_temporal_arcade",
            "superior_arcade",
            "inferior_arcade",
        ]
        disk_regions = ["optic_disk_nasal"]

        if region in central_regions:
            position_weight = 1.5
        elif region in disk_regions:
            position_weight = 1.3
        elif region in vascular_regions:
            position_weight = 1.2
        elif region in peripheral_regions:
            position_weight = 0.8
        else:
            position_weight = 1.0

        return round(base_score * grade_multiplier * position_weight, 4)

    def _get_region_anatomical_info(self, region: str) -> tuple[str, str]:
        """Get anatomical significance and clinical relevance for a region."""
        if region in self.REGION_CLINICAL_RELEVANCE:
            info = self.REGION_CLINICAL_RELEVANCE[region]
            return info["significance"], info.get(
                "high_contribution", info.get("moderate_contribution", "")
            )

        return (
            "Anatomical region of the fundus",
            "Region highlighted by model activation",
        )

    def explain_imaging_prediction(
        self,
        regions: dict[str, list[str | dict]],
        prediction_grade: int,
        confidence: float,
    ) -> ImagingExplanation:
        """Generate feature importance from GradCAM regions for imaging predictions.

        Uses real GradCAM saliency scores when available in region dicts.
        Falls back to synthetic grade-based contribution when only names are provided.

        Args:
            regions: Dictionary with 'left_eye' and 'right_eye' lists of region names or dicts
            prediction_grade: DR grade (0-4)
            confidence: Model confidence score (0-1)

        Returns:
            ImagingExplanation with region-based feature contributions
        """
        left_regions = regions.get("left_eye", [])
        right_regions = regions.get("right_eye", [])

        # Extract unique region names with their actual GradCAM saliency
        region_map: dict[str, float] = {}
        for r in left_regions + right_regions:
            if isinstance(r, dict):
                name = r.get("name", "")
                if name:
                    saliency = r.get("saliency_score", 0.0)
                    intensity = r.get("intensity", 0.0)
                    # Use saliency if available, otherwise intensity, otherwise synthetic
                    current = region_map.get(name, 0.0)
                    region_map[name] = max(current, saliency or intensity)
            elif isinstance(r, str):
                if r not in region_map:
                    region_map[r] = 0.0  # will fill with synthetic below

        all_regions = list(region_map.keys()) or ["posterior_pole"]

        region_contributions = []

        for region in all_regions:
            actual_saliency = region_map.get(region, 0.0)
            if actual_saliency > 0.0:
                contribution = round(actual_saliency, 4)
            else:
                contribution = self._get_region_contribution(
                    region, prediction_grade, confidence
                )
            significance, clinical_relevance = self._get_region_anatomical_info(region)

            region_contributions.append(
                ImagingFeatureContribution(
                    region_name=region,
                    contribution=contribution,
                    anatomical_significance=significance,
                    clinical_relevance=clinical_relevance,
                )
            )

        region_contributions.sort(key=lambda x: x.contribution, reverse=True)

        logger.info(
            f"Imaging explanation computed: {len(region_contributions)} regions, "
            f"grade={prediction_grade}, confidence={confidence}"
        )

        return ImagingExplanation(
            model_type="gradcam_efficientnet",
            prediction_grade=prediction_grade,
            confidence=confidence,
            regions=region_contributions,
            pipeline="imaging",
        )

    async def _load_clinical_model(self) -> Any:
        try:
            model_path = await settings.ensure_clinical_model()
        except RuntimeError as e:
            raise ShapExplainabilityError(str(e))

        import pickle

        def _load(path: Path) -> Any:
            with open(path, "rb") as f:
                return pickle.load(f)

        return await anyio.to_thread.run_sync(_load, model_path, limiter=_IO_LIMITER)

    def _get_feature_names(self) -> list[str]:
        return [
            "thickness_center_fovea",
            "thickness_average_thickness",
            "thickness_total_volume_mm3",
            "thickness_inner_superior",
            "thickness_inner_nasal",
            "thickness_inner_inferior",
            "thickness_inner_temporal",
            "thickness_outer_superior",
            "thickness_outer_nasal",
            "thickness_outer_inferior",
            "thickness_outer_temporal",
            "patient_age",
            "patient_gender",
            "meta_eye",
            "clinical_edema",
            "clinical_erm_status",
            "meta_image_quality",
        ]

    def _encode_features(self, features: dict[str, Any]) -> list[float]:
        feature_names = self._get_feature_names()
        encoded = []

        def to_float(v):
            if v is None:
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, list):
                if len(v) > 0:
                    return to_float(v[0])
                return 0.0
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return 0.0
            return 0.0

        for fname in feature_names:
            if fname in features:
                val = features[fname]
                if isinstance(val, (int, float)):
                    encoded.append(float(val))
                elif isinstance(val, list):
                    if len(val) > 0 and isinstance(val[0], (int, float)):
                        encoded.append(float(val[0]))
                    else:
                        encoded.append(0.0)
                elif isinstance(val, str):
                    if fname == "patient_gender":
                        encoded.append(1.0 if val == "M" else 0.0)
                    elif fname == "meta_eye":
                        encoded.append(1.0 if val == "OD" else 0.0)
                    elif fname == "clinical_edema":
                        encoded.append(1.0 if val == "True" else 0.0)
                    elif fname == "clinical_erm_status":
                        if val == "present":
                            encoded.append(0.0)
                        elif val == "residual":
                            encoded.append(1.0)
                        else:
                            encoded.append(2.0)
                    elif fname == "meta_image_quality":
                        try:
                            encoded.append(float(val))
                        except ValueError:
                            encoded.append(0.0)
                    else:
                        encoded.append(0.0)
                else:
                    encoded.append(to_float(val))
            else:
                encoded.append(0.0)

        return encoded

    async def explain_prediction(
        self,
        features: dict[str, Any],
        pipeline: str = "clinical",
    ) -> ShapExplanation:
        try:
            import shap
        except ImportError:
            raise ShapExplainabilityError("shap package not installed")

        model = await self._load_clinical_model()
        feature_names = self._get_feature_names()
        feature_values = self._encode_features(features)

        feature_array = np.array([feature_values])

        expected_num_features = getattr(model, "n_features_in_", None)
        if expected_num_features is None:
            raise ShapExplainabilityError(
                "Cannot determine expected feature count from model.n_features_in_"
            )

        if len(feature_values) != expected_num_features:
            raise ShapExplainabilityError(
                f"Feature count mismatch: expected {expected_num_features}, got {len(feature_values)}"
            )

        try:
            if hasattr(model, "predict_proba"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(feature_array)

                shap_arr = np.array(shap_values)
                if shap_arr.ndim > 1:
                    shap_values = float(shap_arr.flatten()[0])
                elif shap_arr.ndim == 1:
                    shap_values = shap_arr.tolist()
                else:
                    shap_values = float(shap_arr)

                expected_value = explainer.expected_value
                if expected_value is None:
                    expected_value = 0.5
                elif isinstance(expected_value, (np.ndarray, list, tuple)):
                    expected_value = (
                        float(expected_value[0]) if len(expected_value) > 0 else 0.5
                    )
                else:
                    expected_value = float(expected_value)

            else:
                raise ShapExplainabilityError("Model does not support SHAP")

        except Exception as e:
            logger.warning(f"SHAP calculation failed, using fallback: {e}")
            expected_value = 0.5
            contributions = []
            for fname, fvalue in zip(feature_names, feature_values):
                contributions.append(
                    FeatureContribution(
                        feature_name=fname,
                        contribution=0.0,
                        base_value=expected_value,
                        value=fvalue,
                    )
                )
            return ShapExplanation(
                model_type="xgboost",
                expected_value=expected_value,
                contributions=contributions,
                pipeline=pipeline,
            )

        contributions = []
        shap_vals_list = shap_values if isinstance(shap_values, list) else [shap_values]
        for fname, fvalue, svalue in zip(feature_names, feature_values, shap_vals_list):
            contributions.append(
                FeatureContribution(
                    feature_name=fname,
                    contribution=float(svalue),
                    base_value=expected_value,
                    value=float(fvalue),
                )
            )

        return ShapExplanation(
            model_type="xgboost",
            expected_value=expected_value,
            contributions=contributions,
            pipeline=pipeline,
        )

    async def compute_global_importance(
        self,
        test_csv: Path,
        pipeline: str = "clinical",
        sample_size: int = 100,
    ) -> dict[str, float]:
        try:
            import shap
        except ImportError:
            raise ShapExplainabilityError("shap package not installed")

        import pandas as pd

        model = await self._load_clinical_model()
        df = pd.read_csv(test_csv)

        feature_names = self._get_feature_names()
        available_cols = [c for c in feature_names if c in df.columns]

        if len(available_cols) == 0:
            return {}

        X = df[available_cols].head(sample_size).values
        feature_names = available_cols

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            else:
                shap_values = shap_values

            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            importance = {}
            for fname, mval in zip(feature_names, mean_abs_shap):
                importance[fname] = float(mval)

            self._global_importance[pipeline] = importance
            logger.info(
                f"Computed global SHAP importance for {len(importance)} features"
            )

            return importance

        except Exception as e:
            logger.warning(f"Global SHAP calculation failed: {e}")
            return {}

    def get_global_importance(self, pipeline: str = "clinical") -> dict[str, float]:
        return self._global_importance.get(pipeline, {})

    async def check_bias(
        self,
        test_csv: Path,
        demographic_col: str = "patient_gender",
        pipeline: str = "clinical",
    ) -> dict[str, Any]:
        import pandas as pd

        df = pd.read_csv(test_csv)

        if demographic_col not in df.columns:
            return {"error": f"Demographic column not found: {demographic_col}"}

        groups = df[demographic_col].unique()

        if len(groups) < 2:
            return {"error": "Not enough demographic groups for comparison"}

        bias_results = {}

        for group in groups:
            group_df = df[df[demographic_col] == group]
            if len(group_df) > 10:
                importance = await self.compute_global_importance(test_csv, pipeline)
                bias_results[str(group)] = importance

        if len(bias_results) < 2:
            return {"error": "Insufficient data for bias check"}

        comparison = {}
        for feature in bias_results[list(bias_results.keys())[0]]:
            values = [v.get(feature, 0) for v in bias_results.values()]
            if all(v is not None for v in values):
                diff = max(values) - min(values)
                comparison[feature] = {
                    "max_diff": float(diff),
                    "potentially_biased": diff > 0.1,
                }

        return comparison


_shap_service: ShapService | None = None


def get_shap_service() -> ShapService:
    """FastAPI dependency factory. Creates instance if not overridden."""
    global _shap_service
    if _shap_service is None:
        _shap_service = ShapService()
    return _shap_service
