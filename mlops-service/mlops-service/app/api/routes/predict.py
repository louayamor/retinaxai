from __future__ import annotations

import asyncio
import base64
import io
import math
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_settings
from app.api.schemas import ClinicalFeatures, MLPredictHttpRequest, PredictResponse
from app.config.settings import Settings
from app.services.inference.inference_service import (
    DR_SEVERITY,
    InferenceService,
)
from app.services.platform.websocket_client import (
    send_prediction_event,
    send_prediction_log,
)
from app.services.monitoring.prometheus_metrics import (
    PREDICTION_REQUESTS_TOTAL,
    PREDICTION_ERRORS_TOTAL,
    GRADCAM_GENERATION_FAILURES,
)

router = APIRouter()

MLOPS_BACKEND_WS_URL = "ws://localhost:8000/ws"
MLOPS_BACKEND_API_KEY = ""


def get_inference_service(
    settings: Settings = Depends(get_settings),
) -> InferenceService:
    return InferenceService(settings)


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
    PREDICTION_REQUESTS_TOTAL.labels(model=request.model_name).inc()
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
        await log_msg(
            "step_5",
            "info",
            f"Processing for patient (gender: {request.patient_gender}, age: {request.patient_age})",
        )

        await log_msg("step_6", "info", "Running EfficientNet-B4 for left eye")
        left_imaging_result = await run_in_threadpool(
            service.predict_imaging_with_gradcam, left_bytes, "left"
        )
        await log_msg("step_7", "info", "Running EfficientNet-B4 for right eye")
        right_imaging_result = await run_in_threadpool(
            service.predict_imaging_with_gradcam, right_bytes, "right"
        )
        await log_msg(
            "step_8",
            "success",
            f"Left eye: DR Grade {left_imaging_result['predicted_grade']} ({(left_imaging_result['confidence'] * 100):.1f}%)",
        )
        await log_msg(
            "step_9",
            "success",
            f"Right eye: DR Grade {right_imaging_result['predicted_grade']} ({(right_imaging_result['confidence'] * 100):.1f}%)",
        )

        try:
            features = ClinicalFeatures(**request.features)
            await log_msg("step_10", "info", "Processing clinical features")
            clinical_result = await run_in_threadpool(
                service.predict_clinical, features
            )
            await log_msg(
                "step_11",
                "info",
                f"Clinical model: risk score {clinical_result.get('risk_score', 0):.2f}",
            )
        except Exception:
            features = None
            await log_msg(
                "step_10", "warning", "Skipping clinical model (no valid features)"
            )
            clinical_result = {}

        combined_prediction = {
            "left_eye": left_imaging_result,
            "right_eye": right_imaging_result,
            "clinical": {
                "predicted_grade": clinical_result.get("predicted_grade"),
                "predicted_label": clinical_result.get("predicted_label"),
                "risk_score": clinical_result.get("risk_score"),
                "severity": DR_SEVERITY.get(
                    clinical_result.get("predicted_grade", 0), "unknown"
                ),
                "probabilities": clinical_result.get("probabilities"),
            },
            "combined_grade": max(
                left_imaging_result["predicted_grade"],
                right_imaging_result["predicted_grade"],
            ),
            "overall_severity": DR_SEVERITY.get(
                max(
                    left_imaging_result["predicted_grade"],
                    right_imaging_result["predicted_grade"],
                ),
                "unknown",
            ),
        }

        await log_msg(
            "step_12",
            "success",
            f"Combined: DR Grade {combined_prediction['combined_grade']}",
        )

        left_confidence = left_imaging_result.get("confidence", 0.0)
        if (
            not isinstance(left_confidence, (int, float))
            or math.isnan(left_confidence)
            or math.isinf(left_confidence)
        ):
            left_confidence = 0.0
        right_confidence = right_imaging_result.get("confidence", 0.0)
        if (
            not isinstance(right_confidence, (int, float))
            or math.isnan(right_confidence)
            or math.isinf(right_confidence)
        ):
            right_confidence = 0.0
        clinical_confidence = (
            clinical_result.get("risk_score") if clinical_result else None
        )
        if clinical_confidence is not None:
            if (
                not isinstance(clinical_confidence, (int, float))
                or math.isnan(clinical_confidence)
                or math.isinf(clinical_confidence)
            ):
                clinical_confidence = None

        response = PredictResponse(
            prediction=combined_prediction,
            confidence_score=float(left_confidence),
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
            fundus_score_left=left_imaging_result.get("fundus_score"),
            fundus_score_right=right_imaging_result.get("fundus_score"),
        )

        async def _send_event():
            try:
                await send_prediction_event(
                    prediction_id=prediction_id,
                    patient_id=patient_id,
                    dr_grade=combined_prediction["combined_grade"],
                    confidence=float(left_confidence),
                    imaging_confidence=float(left_confidence),
                    clinical_confidence=clinical_confidence,
                    combined_grade=combined_prediction["combined_grade"],
                    overall_severity=combined_prediction["overall_severity"],
                    triggers_xai=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send prediction event: {e}")

        asyncio.create_task(_send_event())

        try:
            from app.services.platform.feature_store import get_feature_store

            feature_store = get_feature_store()
            feature_store.set(
                f"prediction:{patient_id}:{prediction_id}",
                {
                    "combined_grade": combined_prediction["combined_grade"],
                    "overall_severity": combined_prediction["overall_severity"],
                    "left_grade": left_imaging_result["predicted_grade"],
                    "right_grade": right_imaging_result["predicted_grade"],
                    "left_confidence": left_imaging_result["confidence"],
                    "right_confidence": right_imaging_result["confidence"],
                },
                ttl_seconds=86400,
            )
        except Exception as e:
            logger.warning(f"Failed to cache prediction in feature store: {e}")

        return response

    except ValueError as e:
        PREDICTION_ERRORS_TOTAL.labels(
            model=request.model_name, error_type="fundus_rejection"
        ).inc()
        logger.warning(f"[PREDICT FUNDUS REJECTION] {e}")
        await send_prediction_log(
            patient_id,
            prediction_id,
            "error",
            "error",
            f"Image validation failed: {str(e)[:50]}",
        )
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
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(
            model=request.model_name, error_type=type(e).__name__
        ).inc()
        logger.opt(exception=True).error("[PREDICT ERROR] {}: {}", type(e).__name__, e)
        await send_prediction_log(
            patient_id,
            prediction_id,
            "error",
            "error",
            f"Prediction failed: {type(e).__name__}: {str(e)[:50]}",
        )
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
