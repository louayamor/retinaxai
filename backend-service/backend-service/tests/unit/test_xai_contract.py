from app.api.v1.routes.explanation_routes import XAIResponse, _normalize_risk_level
from app.models.severity_report import RiskLevel


def test_normalize_risk_level_maps_very_high_to_severe() -> None:
    assert _normalize_risk_level("very_high") == RiskLevel.SEVERE


def test_normalize_risk_level_defaults_to_moderate_for_invalid_value() -> None:
    assert _normalize_risk_level("unsupported-level") == RiskLevel.MODERATE


def test_xai_response_accepts_gradcam_explanation_shape() -> None:
    payload = {
        "prediction_id": "d3fc6d95-b2fa-4cb0-a41f-f78dbce57d89",
        "explanation": None,
        "severity_report": None,
        "gradcam_explanation": {
            "id": "c86648fc-f7a4-45ce-a1b0-51cad568a90c",
            "left_eye_explanation": "Left eye heatmap indicates lesions.",
            "right_eye_explanation": "Right eye shows weaker activation.",
            "highlighted_regions": {
                "left_eye": ["macula"],
                "right_eye": ["temporal"],
            },
            "model_used": "gpt-4.1-mini",
        },
    }

    model = XAIResponse.model_validate(payload)

    assert model.gradcam_explanation is not None
    assert model.gradcam_explanation["left_eye_explanation"] == payload[
        "gradcam_explanation"
    ]["left_eye_explanation"]
