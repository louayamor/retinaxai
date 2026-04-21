from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger

from app.schemas import BiomarkerExtractionResponse, VascularBiomarkers
from app.service import BiomarkerService

service = BiomarkerService()


def create_app() -> FastAPI:
    app = FastAPI(title="RetinaXAI Biomarker Service", version=service.service_version)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": service.service_name, "version": service.service_version}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @app.post("/biomarkers/extract", response_model=BiomarkerExtractionResponse)
    async def extract(
        prediction_id: str = Form(...),
        patient_id: str = Form(...),
        image: UploadFile = File(...),
        eye_side: str | None = Form(default=None),
        model_version: str | None = Form(default=None),
    ) -> BiomarkerExtractionResponse:
        try:
            image_bytes = await image.read()
            extracted = service.extract_biomarkers(image_bytes)
            biomarkers = (
                extracted
                if isinstance(extracted, VascularBiomarkers)
                else VascularBiomarkers.model_validate(extracted)
            )
            return BiomarkerExtractionResponse(
                contract_version=service.service_version,
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
        except Exception as exc:
            logger.exception("biomarker extraction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
