from __future__ import annotations

from loguru import logger


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

    REGION_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
        "central": {"regions": ["fovea_centralis", "macula_center", "perifovea", "superior_macula", "inferior_macula", "posterior_pole"], "weight": 1.5},
        "disk": {"regions": ["optic_disk_nasal"], "weight": 1.3},
        "vascular": {"regions": ["temporal_arcade", "superior_temporal_arcade", "inferior_temporal_arcade", "superior_arcade", "inferior_arcade"], "weight": 1.2},
        "peripheral": {"regions": ["nasal_periphery", "temporal_periphery", "superior_periphery", "inferior_periphery", "superior_temporal_periphery", "inferior_temporal_periphery", "superior_nasal_periphery", "inferior_nasal_periphery"], "weight": 0.8},
    }

    GRADE_MULTIPLIERS = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.85, 4: 1.0}

    def __init__(self) -> None:
        pass

    def _get_region_contribution(
        self, region: str, grade: int, confidence: float
    ) -> float:
        base_score = confidence * 0.5
        grade_multiplier = self.GRADE_MULTIPLIERS.get(grade, 0.5)
        position_weight = 1.0
        for group in self.REGION_TYPE_WEIGHTS.values():
            if region in group["regions"]:
                position_weight = group["weight"]
                break
        return round(base_score * grade_multiplier * position_weight, 4)

    def _get_region_anatomical_info(self, region: str) -> tuple[str, str]:
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
        left_regions = regions.get("left_eye", [])
        right_regions = regions.get("right_eye", [])

        region_map: dict[str, float] = {}
        for r in left_regions + right_regions:
            if isinstance(r, dict):
                name = r.get("name", "")
                if name:
                    saliency = r.get("saliency_score", 0.0)
                    intensity = r.get("intensity", 0.0)
                    current = region_map.get(name, 0.0)
                    region_map[name] = max(current, saliency or intensity)
            elif isinstance(r, str):
                if r not in region_map:
                    region_map[r] = 0.0

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


_shap_service: ShapService | None = None


def get_shap_service() -> ShapService:
    global _shap_service
    if _shap_service is None:
        _shap_service = ShapService()
    return _shap_service
