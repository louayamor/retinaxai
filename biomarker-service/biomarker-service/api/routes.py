"""
API routes for the Biomarker Service.

This module defines the FastAPI endpoints for the biomarker extraction service.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse

from biomarker_service.application.orchestrator import BiomarkerOrchestrator
from biomarker_service.application.validators import validate_extraction_request
from biomarker_service.domain.contracts import (
    BiomarkerExtractionRequest,
    BiomarkerExtractionResponse,
)
from biomarker_service.domain.models import BiomarkerFailure
from biomarker_service.infrastructure.model_registry import VascXRegistry

router = APIRouter()

# Initialize orchestrator and registry
orchestrator = BiomarkerOrchestrator()
registry = VascXRegistry()


@router.get("/health")
async def health() -> JSONResponse:
    """
    Health check endpoint.
    """
    return JSONResponse(
        {"status": "ok", "service": "biomarker-service", "version": "0.1.0"}
    )


@router.get("/ready")
async def ready() -> JSONResponse:
    """
    Readiness check endpoint.
    """
    if not registry.is_loaded:
        return JSONResponse({"status": "not ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@router.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/biomarkers/extract", response_model=BiomarkerExtractionResponse)
async def extract(
    prediction_id: str = Form(...),
    patient_id: str = Form(...),
    image: UploadFile = File(...),
    eye_side: str | None = Form(default=None),
    model_version: str | None = Form(default=None),
) -> BiomarkerExtractionResponse:
    """
    Extract vascular biomarkers from a fundus image.
    """
    # Validate request parameters
    try:
        validate_extraction_request(prediction_id, patient_id, eye_side, model_version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Read image bytes
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image payload")

    try:
        # Extract biomarkers
        result = await orchestrator.extract_biomarkers(
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            image_bytes=image_bytes,
        )

        # Return success response
        return BiomarkerExtractionResponse(
            contract_version="1.0",
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            status="success",
            service_name="biomarker-service",
            service_version="0.1.0",
            extracted_at=None,
            biomarkers=result.biomarkers,
            error=None,
        )

    except Exception as exc:
        # Handle failure
        error_code = "BIOMARKER_ERROR"
        error_message = str(exc)
        failure = await orchestrator.handle_failure(
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            error_code=error_code,
            error_message=error_message,
        )

        # Return failure response
        return BiomarkerExtractionResponse(
            contract_version="1.0",
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            status="failed",
            service_name="biomarker-service",
            service_version="0.1.0",
            extracted_at=None,
            biomarkers={},
            error=failure.error_message,
        )


def create_api_router() -> APIRouter:
    """
    Create and return the API router.
    """
    return router
