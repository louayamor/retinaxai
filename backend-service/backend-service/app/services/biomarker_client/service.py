from __future__ import annotations

import logging
from pathlib import Path

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class BiomarkerExtractionError(RuntimeError):
    pass


class BiomarkerServiceClient:
    def __init__(self) -> None:
        self.base_url = settings.BIOMARKER_SERVICE_URL
        self.timeout = settings.BIOMARKER_SERVICE_TIMEOUT
        self.headers = {}
        if settings.BIOMARKER_SERVICE_API_KEY:
            self.headers["Authorization"] = f"Bearer {settings.BIOMARKER_SERVICE_API_KEY}"

    async def extract_from_scan_path(
        self,
        *,
        scan_path: str,
        prediction_id: str,
        patient_id: str,
        eye_side: str,
        model_version: str,
    ) -> dict:
        path = Path(scan_path)
        if not path.exists():
            raise BiomarkerExtractionError(f"scan path not found: {scan_path}")

        try:
            payload = {
                "prediction_id": prediction_id,
                "patient_id": patient_id,
                "eye_side": eye_side,
                "model_version": model_version,
            }
            files = {
                "image": (path.name, path.read_bytes(), "application/octet-stream"),
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/biomarkers/extract",
                    data=payload,
                    files=files,
                    headers=self.headers,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(f"[BIOMARKER CLIENT] HTTP {exc.response.status_code}: {exc.response.text}")
            raise BiomarkerExtractionError(
                f"biomarker service returned {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            logger.error(f"[BIOMARKER CLIENT] extraction failed: {exc}")
            raise BiomarkerExtractionError(str(exc)) from exc


biomarker_client = BiomarkerServiceClient()
