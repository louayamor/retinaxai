from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.metrics import (
    EXTRACTION_DURATION_SECONDS,
    EXTRACTION_FAILURES_TOTAL,
    EXTRACTION_REQUESTS_TOTAL,
)
from app.schemas import (
    BIOMARKER_CONTRACT_VERSION,
    BiomarkerExtractionResponse,
    VascularBiomarkers,
)
from app.service import BiomarkerService

service = BiomarkerService()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read_upload_bytes_with_limit(upload: UploadFile, max_bytes: int) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    total = 0

    await upload.seek(0)
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"Image exceeds maximum size of {max_bytes} bytes")
        chunks.append(chunk)
    await upload.seek(0)
    return b"".join(chunks), total


def create_app() -> FastAPI:
    app = FastAPI(title="RetinaXAI Biomarker Service", version=service.service_version)

    @app.on_event("startup")
    async def _load_vascx() -> None:
        try:
            service.warm()
        except Exception as exc:
            logger.exception("vascx warmup failed: {}", exc)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": service.service_name, "version": service.service_version}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/biomarkers/extract", response_model=BiomarkerExtractionResponse)
    async def extract(
        prediction_id: str = Form(...),
        patient_id: str = Form(...),
        image: UploadFile = File(...),
        eye_side: str | None = Form(default=None),
        model_version: str | None = Form(default=None),
    ) -> BiomarkerExtractionResponse:
        started_at = time.perf_counter()
        logger.info(
            "biomarker extraction request received prediction_id={} patient_id={} eye_side={} model_version={} filename={}",
            prediction_id,
            patient_id,
            eye_side or "unknown",
            model_version or "unknown",
            image.filename or "unknown",
        )
        try:
            if not image.content_type:
                logger.warning(
                    "biomarker extraction missing content_type prediction_id={} patient_id={} eye_side={}",
                    prediction_id,
                    patient_id,
                    eye_side or "unknown",
                )
            else:
                logger.debug(
                    "biomarker extraction content_type={} prediction_id={} patient_id={} eye_side={}",
                    image.content_type,
                    prediction_id,
                    patient_id,
                    eye_side or "unknown",
                )

            image_bytes, image_size = await _read_upload_bytes_with_limit(image, MAX_UPLOAD_BYTES)
            logger.debug(
                "biomarker extraction payload loaded prediction_id={} patient_id={} bytes={}",
                prediction_id,
                patient_id,
                image_size,
            )
            extracted = await asyncio.to_thread(service.extract_biomarkers, image_bytes)
            biomarkers = (
                extracted
                if isinstance(extracted, VascularBiomarkers)
                else VascularBiomarkers.model_validate(extracted)
            )
            response = BiomarkerExtractionResponse(
                contract_version=BIOMARKER_CONTRACT_VERSION,
                prediction_id=prediction_id,
                patient_id=patient_id,
                eye_side=eye_side,
                model_version=model_version,
                status="success",
                service_name=service.service_name,
                service_version=service.service_version,
                extracted_at=datetime.now(timezone.utc),
                biomarkers=biomarkers,
            )
            elapsed = time.perf_counter() - started_at
            EXTRACTION_REQUESTS_TOTAL.labels(status="success").inc()
            EXTRACTION_DURATION_SECONDS.observe(elapsed)
            logger.info(
                "biomarker extraction completed prediction_id={} patient_id={} eye_side={} elapsed_seconds={:.4f}",
                prediction_id,
                patient_id,
                eye_side or "unknown",
                elapsed,
            )
            logger.debug(
                "biomarker extraction response prediction_id={} biomarkers={}",
                prediction_id,
                response.biomarkers.model_dump(),
            )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            EXTRACTION_REQUESTS_TOTAL.labels(status="failed").inc()
            EXTRACTION_FAILURES_TOTAL.labels(reason=type(exc).__name__).inc()
            EXTRACTION_DURATION_SECONDS.observe(elapsed)
            logger.exception(
                "biomarker extraction failed prediction_id={} patient_id={} eye_side={} elapsed_seconds={:.4f}",
                prediction_id,
                patient_id,
                eye_side or "unknown",
                elapsed,
            )
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    return app


app = create_app()
