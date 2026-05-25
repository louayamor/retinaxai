from __future__ import annotations
import base64
from pathlib import Path

import httpx
import structlog

from app.core.config import settings
from app.core.exceptions import (
    ServiceUnavailableException,
    UnprocessableEntityException,
)
from app.services.ml_client.schemas import MLPredictRequest, MLPredictResponse

logger = structlog.get_logger(__name__)


def _encode_image(file_path: str) -> str:
    return base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")


class MLServiceClient:
    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL
        self.timeout = settings.ML_SERVICE_TIMEOUT
        self.headers = {"Content-Type": "application/json"}
        if settings.ML_SERVICE_API_KEY:
            self.headers["Authorization"] = f"Bearer {settings.ML_SERVICE_API_KEY}"

    async def predict(self, request: MLPredictRequest) -> MLPredictResponse:
        logger.info(f"[MLCLIENT] Sending request to {self.base_url}/predict")
        logger.info(
            f"[MLCLIENT] model={request.model_name} version={request.model_version}"
        )
        logger.info(
            f"[MLCLIENT] left_scan_path={request.left_scan_path} exists={Path(request.left_scan_path).exists()}"
        )
        logger.info(
            f"[MLCLIENT] right_scan_path={request.right_scan_path} exists={Path(request.right_scan_path).exists()}"
        )
        logger.info(
            f"[MLCLIENT] patient age={request.patient_age} gender={request.patient_gender}"
        )

        payload = {
            "model_name": request.model_name,
            "model_version": request.model_version,
            "patient_id": request.patient_id,
            "patient_age": request.patient_age,
            "patient_gender": request.patient_gender,
            "left_scan_path": request.left_scan_path,
            "right_scan_path": request.right_scan_path,
            "features": request.features,
        }
        payload["left_scan"] = _encode_image(request.left_scan_path)
        payload["right_scan"] = _encode_image(request.right_scan_path)
        logger.info(f"[MLCLIENT] Payload keys: {list(payload.keys())}")
        logger.info(f"[MLCLIENT] left_scan encoded length: {len(payload['left_scan'])}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"[MLCLIENT] POSTing to {self.base_url}/predict")
                response = await client.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    headers=self.headers,
                )
                logger.info(f"[MLCLIENT] Response status={response.status_code}")
                if response.status_code == 422:
                    logger.error(f"[MLCLIENT] 422 from MLOps: {response.text}")
                    raise UnprocessableEntityException(
                        response.json().get("detail", "Invalid input payload.")
                    )
                response.raise_for_status()
                return MLPredictResponse(**response.json())
        except httpx.TimeoutException:
            logger.error(
                f"[MLCLIENT] Timeout connecting to ML service at {self.base_url}"
            )
            raise ServiceUnavailableException("ml-service")
        except httpx.ConnectError as e:
            logger.error(
                f"[MLCLIENT] Connection error to ML service at {self.base_url}: {e}"
            )
            raise ServiceUnavailableException("ml-service")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[MLCLIENT] HTTP error {e.response.status_code}: {e.response.text}"
            )
            raise


ml_client = MLServiceClient()
