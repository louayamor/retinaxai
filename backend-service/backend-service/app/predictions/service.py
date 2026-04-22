import asyncio
import logging
import traceback
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException
from app.models.prediction import Prediction, PredictionStatus
from app.models.vascular_biomarker import BiomarkerStatus
from app.mri_scans.repository import MRIScanRepository
from app.patients.repository import PatientRepository
from app.predictions.repository import PredictionRepository
from app.schemas.prediction_schema import PredictionRequest
from app.services.biomarker_client.service import biomarker_client
from app.services.ml_client.ml_service import ml_client
from app.services.ml_client.schemas import MLPredictRequest
from app.websockets.manager import get_socket_manager

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = PredictionRepository(db)
        self.patient_repo = PatientRepository(db)
        self.mri_scan_repo = MRIScanRepository(db)

    async def run(self, data: PredictionRequest, requested_by: uuid.UUID) -> Prediction:
        logger.info(
            f"[PREDICT SERVICE] Starting prediction for patient={data.patient_id} scan={data.mri_scan_id}"
        )

        patient = await self.patient_repo.get_by_id(data.patient_id)
        if not patient:
            raise NotFoundException("Patient", data.patient_id)  # type: ignore[reportArgumentType]

        scan = await self.mri_scan_repo.get_by_id(data.mri_scan_id)
        if not scan:
            raise NotFoundException("MRIScan", data.mri_scan_id)  # type: ignore[reportArgumentType]

        logger.info(
            f"[PREDICT SERVICE] Patient found: {patient.first_name} age={patient.age} gender={patient.gender}"
        )
        logger.info(
            f"[PREDICT SERVICE] Scan found: left={scan.left_scan_path} right={scan.right_scan_path}"
        )

        prediction = Prediction(
            patient_id=data.patient_id,
            mri_scan_id=data.mri_scan_id,
            requested_by=requested_by,
            model_name=data.model_name,
            model_version=data.model_version,
            input_payload=data.input_payload,
            status=PredictionStatus.PENDING,
        )
        prediction = await self.repo.create(prediction)
        logger.info(f"[PREDICT SERVICE] Prediction record created: {prediction.id}")

        try:
            ml_request = MLPredictRequest(
                model_name=data.model_name,
                model_version=data.model_version,
                patient_id=str(data.patient_id),
                patient_age=patient.age,
                patient_gender=patient.gender.value,
                left_scan_path=scan.left_scan_path,
                right_scan_path=scan.right_scan_path,
                features=data.input_payload,
            )
            logger.info("[PREDICT SERVICE] Calling ml_client.predict()")
            ml_response = await ml_client.predict(ml_request)
            logger.info(
                f"[PREDICT SERVICE] ml_client.predict() succeeded: confidence={ml_response.confidence_score}"
            )

            output_payload = {**ml_response.prediction}
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
            if ml_response.shap_explanation:
                output_payload["shap_explanation"] = ml_response.shap_explanation

            prediction.output_payload = output_payload
            prediction.confidence_score = ml_response.confidence_score
            prediction.status = PredictionStatus.SUCCESS
            logger.info(f"[PREDICT SERVICE] Prediction SUCCESS: {prediction.id}")

            dr_grade = output_payload.get("combined_grade", 0)
            severity_map = {
                0: "none",
                1: "low",
                2: "moderate",
                3: "high",
                4: "critical",
            }
            overall_severity = severity_map.get(dr_grade, "unknown")

            try:
                socket_manager = get_socket_manager()
                asyncio.create_task(
                    socket_manager.emit_prediction_event(
                        prediction_id=str(prediction.id),
                        patient_id=str(data.patient_id),
                        status="completed",
                        dr_grade=dr_grade,
                        confidence=ml_response.confidence_score,
                        overall_severity=overall_severity,
                        triggers_xai=True,
                    )
                )
                logger.info(
                    f"[PREDICT SERVICE] Emitted prediction.completed WebSocket event for {prediction.id}"
                )
            except Exception as ws_error:
                logger.warning(
                    f"[PREDICT SERVICE] Failed to emit WebSocket event: {ws_error}"
                )

            patient_data = {
                "name": f"{patient.first_name} {patient.last_name}",
                "age": patient.age,
                "gender": patient.gender.value,
            }

            try:
                socket_manager = get_socket_manager()
                await socket_manager.emit_biomarker_event(
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="left",
                    status="started",
                    progress=65,
                    message="Extracting left-eye biomarkers...",
                )
                left_biomarker_payload = await biomarker_client.extract_from_scan_path(
                    scan_path=scan.left_scan_path,
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="left",
                    model_version=data.model_version,
                )
                await socket_manager.emit_biomarker_event(
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="left",
                    status="completed",
                    progress=72,
                    message="Left-eye biomarkers extracted",
                    biomarkers=left_biomarker_payload.get("biomarkers", {}),
                )

                await socket_manager.emit_biomarker_event(
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="right",
                    status="started",
                    progress=76,
                    message="Extracting right-eye biomarkers...",
                )
                right_biomarker_payload = await biomarker_client.extract_from_scan_path(
                    scan_path=scan.right_scan_path,
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="right",
                    model_version=data.model_version,
                )
                await socket_manager.emit_biomarker_event(
                    prediction_id=str(prediction.id),
                    patient_id=str(data.patient_id),
                    eye_side="right",
                    status="completed",
                    progress=84,
                    message="Right-eye biomarkers extracted",
                    biomarkers=right_biomarker_payload.get("biomarkers", {}),
                )

                output_payload = dict(prediction.output_payload or {})
                output_payload["vascular_biomarkers_left"] = left_biomarker_payload.get(
                    "biomarkers", {}
                )
                output_payload["vascular_biomarkers_right"] = right_biomarker_payload.get(
                    "biomarkers", {}
                )
                prediction.output_payload = output_payload
                prediction.biomarker_status = BiomarkerStatus.COMPLETED
                prediction.biomarker_error_message = None

                try:
                    from app.explanations.service import ExplanationService

                    explain_service = ExplanationService(self.db)
                    patient_data["vascular_biomarkers"] = {
                        "left_eye": left_biomarker_payload.get("biomarkers", {}),
                        "right_eye": right_biomarker_payload.get("biomarkers", {}),
                    }
                    await explain_service.trigger_xai_for_prediction(prediction, patient_data)
                    logger.info(
                        f"[PREDICT SERVICE] XAI pipeline triggered for {prediction.id}"
                    )
                except Exception as e:
                    logger.warning(f"[PREDICT SERVICE] Failed to trigger XAI: {e}")
            except Exception as biomarker_error:
                prediction.biomarker_status = BiomarkerStatus.FAILED
                prediction.biomarker_error_message = str(biomarker_error)[:497]
                try:
                    socket_manager = get_socket_manager()
                    await socket_manager.emit_biomarker_event(
                        prediction_id=str(prediction.id),
                        patient_id=str(data.patient_id),
                        eye_side="both",
                        status="failed",
                        progress=70,
                        message="Biomarker extraction failed",
                        error=str(biomarker_error)[:200],
                    )
                except Exception as ws_error:
                    logger.warning(
                        f"[PREDICT SERVICE] Failed to emit biomarker failure event: {ws_error}"
                    )
                logger.warning(
                    f"[PREDICT SERVICE] Biomarker extraction failed but prediction remains successful: {biomarker_error}"
                )
        except Exception as e:
            logger.error(
                f"[PREDICT SERVICE] Prediction FAILED: {type(e).__name__}: {e}"
            )
            logger.error(f"[PREDICT SERVICE] Traceback: {traceback.format_exc()}")
            prediction.status = PredictionStatus.FAILED
            error_str = f'{type(e).__name__}: {str(e)}'
            prediction.error_message = error_str[:497] if len(error_str) <= 497 else error_str[:497] + '...'

            try:
                socket_manager = get_socket_manager()
                asyncio.create_task(
                    socket_manager.emit_prediction_event(
                        prediction_id=str(prediction.id),
                        patient_id=str(data.patient_id),
                        status="failed",
                        dr_grade=0,
                        confidence=0.0,
                        overall_severity="unknown",
                        triggers_xai=False,
                        error=str(e)[:200],
                    )
                )
                logger.info(
                    f"[PREDICT SERVICE] Emitted prediction.failed WebSocket event for {prediction.id}"
                )
            except Exception as ws_error:
                logger.warning(
                    f"[PREDICT SERVICE] Failed to emit WebSocket failure event: {ws_error}"
                )

        return await self.repo.update(prediction)

    async def get_by_id(self, prediction_id: uuid.UUID) -> Prediction:
        prediction = await self.repo.get_by_id(prediction_id)
        if not prediction:
            raise NotFoundException("Prediction", prediction_id)  # type: ignore[reportArgumentType]
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
