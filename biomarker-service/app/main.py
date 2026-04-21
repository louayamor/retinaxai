from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger

from app.schemas import BiomarkerExtractionResponse
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
        prediction_id: str,
        patient_id: str,
        image: UploadFile = File(...),
        eye_side: str | None = None,
        model_version: str | None = None,
    ) -> BiomarkerExtractionResponse:
        try:
            image_bytes = await image.read()
            biomarkers = service.extract_biomarkers(image_bytes)
            return BiomarkerExtractionResponse(
                prediction_id=prediction_id,
                status="success",
                service_name=service.service_name,
                service_version=service.service_version,
                biomarkers=biomarkers,
            )
        except Exception as exc:
            logger.exception("biomarker extraction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
