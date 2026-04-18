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
from app.services.platform.websocket_client import send_prediction_event

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
    try:
        logger.info(f"[STEP 1] Decoding left image ({len(request.left_scan)} chars)")
        left_bytes = _decode_base64_image(request.left_scan)
        logger.info(f"[STEP 2] Decoding right image ({len(request.right_scan)} chars)")
        right_bytes = _decode_base64_image(request.right_scan)
        logger.info("[STEP 3] Validating left image")
        _validate_image_bytes(left_bytes)
        logger.info("[STEP 4] Validating right image")
        _validate_image_bytes(right_bytes)
        logger.info(
            f"[STEP 5] Processing prediction for {request.patient_gender} "
            f"age {request.patient_age}, model={request.model_name} v{request.model_version}"
        )

        logger.info("[STEP 6] Loading/predicting imaging model with GradCAM (left eye)")
        left_imaging_result = service.predict_imaging_with_gradcam(left_bytes)
        logger.info(
            "[STEP 7] Loading/predicting imaging model with GradCAM (right eye)"
        )
        right_imaging_result = service.predict_imaging_with_gradcam(right_bytes)
        logger.info(f"[STEP 8] Left eye result: {left_imaging_result}")
        logger.info(f"[STEP 9] Right eye result: {right_imaging_result}")

        logger.info(f"[STEP 10] Parsing clinical features: {request.features}")
        try:
            features = ClinicalFeatures(**request.features)
        except Exception as e:
            logger.warning(
                f"[STEP 10] Clinical features parse failed: {e} — proceeding without clinical model"
            )
            features = None

        if features is not None:
            logger.info("[STEP 11] Loading/predicting clinical model")
            clinical_result = service.predict_clinical(features)
            logger.info(f"[STEP 12] Clinical result: {clinical_result}")
        else:
            logger.info("[STEP 11-12] Skipping clinical model (no valid features)")
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

        logger.info(
            f"prediction complete: left={left_imaging_result['predicted_grade']} "
            f"right={right_imaging_result['predicted_grade']} "
            f"combined={combined_prediction['combined_grade']} "
            f"left_confidence={left_imaging_result['confidence']}"
        )

        response = PredictResponse(
            prediction=combined_prediction,
            confidence_score=left_imaging_result["confidence"],
            model_name=request.model_name,
            model_version=request.model_version,
            gradcam_left=left_imaging_result.get("gradcam_heatmap"),
            gradcam_right=right_imaging_result.get("gradcam_heatmap"),
            regions_left=left_imaging_result.get("regions"),
            regions_right=right_imaging_result.get("regions"),
            shap_explanation=None,
        )

        prediction_id = (
            f"pred_{request.patient_id}_{int(datetime.utcnow().timestamp())}"
        )

        asyncio.create_task(
            send_prediction_event(
                prediction_id=prediction_id,
                patient_id=request.patient_id,
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
        logger.error(f"[PREDICT ERROR] {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}"
        )
