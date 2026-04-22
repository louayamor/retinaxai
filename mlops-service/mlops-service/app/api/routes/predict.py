import asyncio
import base64
import io
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from PIL import Image

from app.api.dependencies import get_settings
from app.api.schemas import ClinicalFeatures, MLPredictHttpRequest, PredictResponse
from app.config.settings import Settings
from app.services.inference.inference_service import InferenceService
from app.services.platform.websocket_client import send_prediction_event, send_prediction_log

router = APIRouter()
_inference_service = None

MLOPS_BACKEND_WS_URL = "ws://localhost:8000/ws"
MLOPS_BACKEND_API_KEY = ""


def get_inference_service(
    settings: Settings = Depends(get_settings),
) -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(settings)
    return _inference_service


def _decode_base64_image(base64_str: str) -> bytes:
    try:
        return base64.b64decode(base64_str, validate=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid base64 image: {e}")


def _validate_image_bytes(image_bytes: bytes) -> None:
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid image data: {e}")


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: MLPredictHttpRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    patient_id = request.patient_id
    prediction_id = f"pred_{patient_id}_{int(datetime.utcnow().timestamp())}"

    async def log_msg(step: str, status: str, message: str):
        await send_prediction_log(patient_id, prediction_id, step, status, message)

    try:
        await log_msg("step_1", "info", "Decoding left fundus image")
        left_bytes = _decode_base64_image(request.left_scan)
        await log_msg("step_2", "info", "Decoding right fundus image")
        right_bytes = _decode_base64_image(request.right_scan)
        await log_msg("step_3", "info", "Validating left image")
        _validate_image_bytes(left_bytes)
        await log_msg("step_4", "info", "Validating right image")
        _validate_image_bytes(right_bytes)
        await log_msg("step_5", "info", f"Processing for patient (gender: {request.patient_gender}, age: {request.patient_age})")

        await log_msg("step_6", "info", "Running EfficientNet-B3 for left eye")
        left_imaging_result = service.predict_imaging_with_gradcam(left_bytes)
        await log_msg("step_7", "info", "Running EfficientNet-B3 for right eye")
        right_imaging_result = service.predict_imaging_with_gradcam(right_bytes)
        await log_msg("step_8", "success", f"Left eye: DR Grade {left_imaging_result['predicted_grade']} ({(left_imaging_result['confidence']*100):.1f}%)")
        await log_msg("step_9", "success", f"Right eye: DR Grade {right_imaging_result['predicted_grade']} ({(right_imaging_result['confidence']*100):.1f}%)")

        try:
            features = ClinicalFeatures(**request.features)
            await log_msg("step_10", "info", "Processing clinical features")
            clinical_result = service.predict_clinical(features)
            await log_msg("step_11", "info", f"Clinical model: risk score {clinical_result.get('risk_score', 0):.2f}")
        except Exception:
            features = None
            await log_msg("step_10", "warning", "Skipping clinical model (no valid features)")
            clinical_result = {}

        severity_map = {0: "none", 1: "low", 2: "moderate", 3: "high", 4: "critical"}

        combined_prediction = {
            "left_eye": left_imaging_result,
            "right_eye": right_imaging_result,
            "clinical": {
                "predicted_grade": clinical_result.get("predicted_grade"),
                "predicted_label": clinical_result.get("predicted_label"),
                "risk_score": clinical_result.get("risk_score"),
                "severity": severity_map.get(
                    clinical_result.get("predicted_grade", 0), "unknown"
                ),
                "probabilities": clinical_result.get("probabilities"),
            },
            "combined_grade": max(
                left_imaging_result["predicted_grade"],
                right_imaging_result["predicted_grade"],
            ),
            "overall_severity": severity_map.get(
                max(
                    left_imaging_result["predicted_grade"],
                    right_imaging_result["predicted_grade"],
                ),
                "unknown",
            ),
        }

        await log_msg("step_12", "success", f"Combined: DR Grade {combined_prediction['combined_grade']}")

        response = PredictResponse(
            prediction=combined_prediction,
            confidence_score=left_imaging_result["confidence"],
            model_name=request.model_name,
            model_version=request.model_version,
            embedding=left_imaging_result.get("embedding"),
            gradcam_left=left_imaging_result.get("gradcam_heatmap"),
            gradcam_right=right_imaging_result.get("gradcam_heatmap"),
            regions_left=left_imaging_result.get("regions"),
            regions_right=right_imaging_result.get("regions"),
            top_hotspots_left=left_imaging_result.get("top_hotspots"),
            top_hotspots_right=right_imaging_result.get("top_hotspots"),
            shap_explanation=None,
        )

        asyncio.create_task(
            send_prediction_event(
                prediction_id=prediction_id,
                patient_id=patient_id,
                dr_grade=combined_prediction["combined_grade"],
                confidence=left_imaging_result["confidence"],
                imaging_confidence=left_imaging_result["confidence"],
                clinical_confidence=clinical_result.get("risk_score"),
                combined_grade=combined_prediction["combined_grade"],
                overall_severity=combined_prediction["overall_severity"],
                triggers_xai=True,
            )
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.opt(exception=True).error("[PREDICT ERROR] {}: {}", type(e).__name__, e)
        await send_prediction_log(patient_id, prediction_id, "error", "error", f"Prediction failed: {type(e).__name__}: {str(e)[:50]}")
        await send_prediction_event(
            prediction_id=prediction_id,
            patient_id=patient_id,
            dr_grade=0,
            confidence=0.0,
            imaging_confidence=0.0,
            clinical_confidence=None,
            combined_grade=0,
            overall_severity="unknown",
            triggers_xai=False,
            error=str(e)[:200],
        )
        raise HTTPException(
            status_code=500, detail=f"prediction failed: {type(e).__name__}"
        )
