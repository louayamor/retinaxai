from __future__ import annotations
import traceback
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException, UnprocessableEntityException
from app.models.patient import Patient
from app.models.mri_scan import MRIScan
from app.models.prediction import Prediction, PredictionStatus
from app.mri_scans.repository import MRIScanRepository
from app.patients.repository import PatientRepository
from app.predictions.repository import PredictionRepository
from app.schemas.prediction_schema import PredictionRequest
from app.services.ml_client.ml_service import MLServiceClient, ml_client
from app.services.ml_client.schemas import MLPredictRequest, MLPredictResponse

from app.api.v1.websockets import emit_prediction_event
from app.explanations.service import ExplanationService
from app.services.task_tracker import bg_tasks

from loguru import logger

_SEVERITY_MAP: dict[int, str] = {
    0: "none",
    1: "low",
    2: "moderate",
    3: "high",
    4: "critical",
}


class PredictionService:
    def __init__(
        self,
        db: AsyncSession,
        ml_client_override: MLServiceClient | None = None,
    ):
        self.db = db
        self.repo = PredictionRepository(db)
        self.patient_repo = PatientRepository(db)
        self.mri_scan_repo = MRIScanRepository(db)
        self._ml_client = ml_client_override or ml_client

    async def _get_patient_and_scan(
        self, patient_id: uuid.UUID, mri_scan_id: uuid.UUID
    ) -> tuple[Patient, MRIScan]:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", patient_id)
        scan = await self.mri_scan_repo.get_by_id(mri_scan_id)
        if not scan:
            raise NotFoundException("MRIScan", mri_scan_id)
        return patient, scan

    async def _create_prediction(
        self, data: PredictionRequest, requested_by: uuid.UUID
    ) -> Prediction:
        prediction = Prediction(
            patient_id=data.patient_id,
            mri_scan_id=data.mri_scan_id,
            requested_by=requested_by,
            model_name=data.model_name,
            model_version=data.model_version,
            input_payload=data.input_payload,
            status=PredictionStatus.PENDING,
        )
        return await self.repo.create(prediction)

    def _build_ml_payload(
        self,
        data: PredictionRequest,
        patient: Patient,
        scan: MRIScan,
    ) -> MLPredictRequest:
        return MLPredictRequest(
            model_name=data.model_name,
            model_version=data.model_version,
            patient_id=str(data.patient_id),
            patient_age=patient.age,
            patient_gender=patient.gender.value,
            left_scan_path=scan.left_scan_path,
            right_scan_path=scan.right_scan_path,
            features=data.input_payload,
        )

    def _merge_ml_response(
        self, ml_response: MLPredictResponse
    ) -> dict[str, Any]:
        output_payload: dict[str, Any] = {**ml_response.prediction}
        if ml_response.embedding:
            output_payload["embedding"] = ml_response.embedding
        if ml_response.gradcam_left:
            output_payload["gradcam_left"] = ml_response.gradcam_left
        if ml_response.gradcam_right:
            output_payload["gradcam_right"] = ml_response.gradcam_right
        if ml_response.regions_left:
            output_payload["gradcam_left_regions"] = [
                r.model_dump() for r in ml_response.regions_left
            ]
        if ml_response.regions_right:
            output_payload["gradcam_right_regions"] = [
                r.model_dump() for r in ml_response.regions_right
            ]
        if ml_response.top_hotspots_left:
            output_payload["top_hotspots_left"] = [
                h.model_dump() for h in ml_response.top_hotspots_left
            ]
        if ml_response.top_hotspots_right:
            output_payload["top_hotspots_right"] = [
                h.model_dump() for h in ml_response.top_hotspots_right
            ]
        # Lesion information (if present)
        if getattr(ml_response, "lesions_left", None):
            output_payload["lesions_left"] = ml_response.lesions_left
        if getattr(ml_response, "lesions_right", None):
            output_payload["lesions_right"] = ml_response.lesions_right
        if getattr(ml_response, "lesion_clusters_left", None):
            output_payload["lesion_clusters_left"] = [
                c for c in ml_response.lesion_clusters_left
            ]
        if getattr(ml_response, "lesion_clusters_right", None):
            output_payload["lesion_clusters_right"] = [
                c for c in ml_response.lesion_clusters_right
            ]
        return output_payload

    async def _handle_prediction_success(
        self,
        prediction: Prediction,
        ml_response: MLPredictResponse,
        data: PredictionRequest,
        patient: Patient,
    ) -> None:
        output_payload = self._merge_ml_response(ml_response)
        prediction.output_payload = output_payload
        prediction.confidence_score = ml_response.confidence_score
        prediction.status = PredictionStatus.SUCCESS

        dr_grade = output_payload.get("combined_grade", 0)
        overall_severity = _SEVERITY_MAP.get(dr_grade, "unknown")

        bg_tasks.create_task(
            emit_prediction_event(
                prediction_id=str(prediction.id),
                patient_id=str(data.patient_id),
                status="completed",
                dr_grade=dr_grade,
                confidence=ml_response.confidence_score,
                overall_severity=overall_severity,
                triggers_xai=True,
            )
        )

        patient_data = {
            "name": f"{patient.first_name} {patient.last_name}",
            "age": patient.age,
            "gender": patient.gender.value,
        }

        try:
            explain_service = ExplanationService(self.db)
            await explain_service.trigger_xai_for_prediction(prediction, patient_data)
        except Exception as e:
            logger.warning(f"[PREDICT SERVICE] Failed to trigger XAI: {e}")

    async def _emit_prediction_failed(
        self, prediction_id: uuid.UUID, patient_id: uuid.UUID, error: str
    ) -> None:
        bg_tasks.create_task(
            emit_prediction_event(
                prediction_id=str(prediction_id),
                patient_id=str(patient_id),
                status="failed",
                dr_grade=0,
                confidence=0.0,
                overall_severity="unknown",
                triggers_xai=False,
                error=error[:200],
            )
        )

    async def run(self, data: PredictionRequest, requested_by: uuid.UUID) -> Prediction:
        logger.info(
            f"[PREDICT SERVICE] Starting prediction for patient={data.patient_id} scan={data.mri_scan_id}"
        )

        patient, scan = await self._get_patient_and_scan(data.patient_id, data.mri_scan_id)
        prediction = await self._create_prediction(data, requested_by)

        try:
            ml_request = self._build_ml_payload(data, patient, scan)
            ml_response = await self._ml_client.predict(ml_request)
            await self._handle_prediction_success(prediction, ml_response, data, patient)
        except UnprocessableEntityException as e:
            await self.repo.delete(prediction)
            raise HTTPException(status_code=422, detail=str(e.detail))
        except Exception as e:
            logger.error(f"[PREDICT SERVICE] Prediction FAILED: {type(e).__name__}: {e}")
            logger.error(f"[PREDICT SERVICE] Traceback: {traceback.format_exc()}")
            prediction.status = PredictionStatus.FAILED
            error_str = f"{type(e).__name__}: {str(e)}"
            prediction.error_message = (
                error_str[:497] if len(error_str) <= 497 else error_str[:497] + "..."
            )
            await self._emit_prediction_failed(prediction.id, data.patient_id, str(e))

        return await self.repo.update(prediction)

    async def get_by_id(self, prediction_id: uuid.UUID) -> Prediction:
        prediction = await self.repo.get_by_id(prediction_id)
        if not prediction:
            raise NotFoundException("Prediction", prediction_id)
        return prediction

    async def get_by_patient(
        self, patient_id: uuid.UUID, skip: int = 0, limit: int = 20
    ) -> tuple[list[Prediction], int]:
        predictions = await self.repo.get_by_patient(patient_id, skip, limit)
        total = await self.repo.count_by_patient(patient_id)
        return predictions, total

    async def get_all(
        self, skip: int = 0, limit: int = 20
    ) -> tuple[list[Prediction], int]:
        predictions = await self.repo.get_all(skip, limit)
        total = await self.repo.count_all()
        return predictions, total
