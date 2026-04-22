from __future__ import annotations

from app.api.routes import XAIPredictionRequest
from app.pipeline.xai_pipeline import XAIPipeline


def test_xai_prediction_request_accepts_vascular_biomarkers() -> None:
    payload = {
        "prediction_id": "pred-1",
        "dr_grade": 3,
        "confidence": 0.91,
        "clinical_features": {"age": 64},
        "gradcam_regions": {"left_eye": ["macula"], "right_eye": ["retina"]},
        "vascular_biomarkers": {
            "left_eye": {"tortuosity": 0.4},
            "right_eye": {"tortuosity": 0.6},
        },
    }

    model = XAIPredictionRequest.model_validate(payload)

    assert model.vascular_biomarkers is not None
    assert model.vascular_biomarkers["left_eye"]["tortuosity"] == 0.4


def test_imaging_prompt_includes_biomarkers() -> None:
    pipeline = XAIPipeline.__new__(XAIPipeline)
    prompt = pipeline._build_imaging_prompt_with_regions(  # type: ignore[attr-defined]
        dr_grade="3",
        confidence=0.91,
        gradcam_regions={"left_eye": ["macula"], "right_eye": ["retina"]},
        shap_values=None,
        vascular_biomarkers={"left_eye": {"tortuosity": 0.4}},
    )

    assert "Vascular Biomarkers" in prompt
    assert "tortuosity" in prompt
