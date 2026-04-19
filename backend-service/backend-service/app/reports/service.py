import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
from app.core.exceptions import (
    ConflictException,
    NotFoundException,
    UnprocessableEntityException,
)
from app.models.prediction import PredictionStatus
from app.models.report import Report, ReportStatus, ReportType
from app.patients.repository import PatientRepository
from app.predictions.repository import PredictionRepository
from app.reports.repository import ReportRepository
from app.schemas.report_schema import OCTReportCreate, ReportGenerateRequest
from app.services.llm_client.llm_service import llm_client
from app.services.llm_client.schemas import LLMReportRequest


class ReportService:
    def __init__(self, db: AsyncSession):
        self.repo = ReportRepository(db)
        self.prediction_repo = PredictionRepository(db)
        self.patient_repo = PatientRepository(db)
        self.db = db

    async def create_oct_report(
        self, data: OCTReportCreate, generated_by: uuid.UUID
    ) -> Report:
        patient = await self.patient_repo.get_by_id(data.patient_id)
        if not patient:
            raise NotFoundException("Patient", data.patient_id)

        report = Report(
            patient_id=data.patient_id,
            prediction_id=None,
            generated_by=generated_by,
            llm_model="ocr-model",
            status=ReportStatus.COMPLETED,
            report_type=ReportType.OCT,
            eye=data.eye,
            source_file=data.source_file,
            dr_grade=data.dr_grade,
            edema=data.edema,
            erm_status=data.erm_status,
            image_quality=data.image_quality,
            thickness_center_fovea=data.thickness_center_fovea,
            thickness_average_thickness=data.thickness_average_thickness,
            thickness_total_volume_mm3=data.thickness_total_volume_mm3,
            thickness_inner_superior=data.thickness_inner_superior,
            thickness_inner_nasal=data.thickness_inner_nasal,
            thickness_inner_inferior=data.thickness_inner_inferior,
            thickness_inner_temporal=data.thickness_inner_temporal,
            thickness_outer_superior=data.thickness_outer_superior,
            thickness_outer_nasal=data.thickness_outer_nasal,
            thickness_outer_inferior=data.thickness_outer_inferior,
            thickness_outer_temporal=data.thickness_outer_temporal,
        )
        return await self.repo.create(report)

    async def get_oct_reports(
        self, skip: int = 0, limit: int = 100
    ) -> tuple[list[Report], int]:
        reports = await self.repo.get_by_type(ReportType.OCT, skip, limit)
        total = await self.repo.count_by_type(ReportType.OCT)
        return reports, total

    async def get_oct_report_by_patient(
        self, patient_id: uuid.UUID
    ) -> list[Report]:
        return await self.repo.get_by_patient_and_type(patient_id, ReportType.OCT)

    async def generate(
        self, data: ReportGenerateRequest, generated_by: uuid.UUID
    ) -> Report:
        prediction = await self.prediction_repo.get_by_id(data.prediction_id)
        if not prediction:
            raise NotFoundException("Prediction", data.prediction_id)  # type: ignore[reportArgumentType]

        if prediction.status != PredictionStatus.SUCCESS:
            raise UnprocessableEntityException(
                "Cannot generate a report for a prediction that did not succeed."
            )

        existing = await self.repo.get_by_prediction_id(data.prediction_id)
        if existing:
            if existing.status == ReportStatus.FAILED:
                await self.db.delete(existing)
                await self.db.flush()
            else:
                raise ConflictException("A report for this prediction already exists.")

        patient = await self.patient_repo.get_by_id(prediction.patient_id)
        if not patient:
            raise NotFoundException("Patient", prediction.patient_id)  # type: ignore[reportArgumentType]

        report = Report(
            patient_id=prediction.patient_id,
            prediction_id=data.prediction_id,
            generated_by=generated_by,
            llm_model=settings.LLM_MODEL,
            status=ReportStatus.GENERATING,
            report_type=ReportType.LLM,
        )
        report = await self.repo.create(report)

        try:
            cleaned_summary = ""
            output_payload = prediction.output_payload or {}

            cleaned_summary = str(
                output_payload.get("summary")
                or output_payload.get("description")
                or output_payload.get("dr_grade", "")
            )

            combined_grade = output_payload.get("combined_grade")
            predicted_class = output_payload.get("predicted_class")
            grade_text = (
                GRADE_LABELS[combined_grade]
                if combined_grade is not None
                else predicted_class or "Unknown"
            )

            prediction_summary = {
                "id": str(prediction.id),
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
                "confidence_score": prediction.confidence_score,
                "status": prediction.status.value,
                "dr_grade": grade_text,
                "severity": output_payload.get("overall_severity", "unknown"),
            }

            llm_request = LLMReportRequest(
                patient={
                    "id": str(patient.id),
                    "first_name": patient.first_name,
                    "last_name": patient.last_name,
                    "age": patient.age,
                    "gender": patient.gender.value,
                    "medical_record_number": patient.medical_record_number,
                },
                prediction=prediction_summary,
                cleaned_summary=cleaned_summary,
                raw_ocr_text="",
                report_type="prediction",
                model_name=prediction.model_name,
                model_version=prediction.model_version,
                prediction_output=prediction_summary,
                confidence_score=prediction.confidence_score
                if prediction.confidence_score is not None
                else 0.0,
            )
            llm_response = await llm_client.generate_report(llm_request)
            report.content = llm_response.content
            report.summary = llm_response.summary
            report.llm_model = llm_response.model_used
            report.status = ReportStatus.COMPLETED
        except Exception as e:
            report.status = ReportStatus.FAILED
            report.error_message = str(e)

        return await self.repo.update(report)

    async def get_by_id(self, report_id: uuid.UUID) -> Report:
        report = await self.repo.get_by_id(report_id)
        if not report:
            raise NotFoundException("Report", report_id)  # type: ignore[reportArgumentType]
        return report

    async def get_by_patient(
        self, patient_id: uuid.UUID, skip: int = 0, limit: int = 20
    ) -> tuple[list[Report], int]:
        reports = await self.repo.get_by_patient(patient_id, skip, limit)
        total = await self.repo.count_by_patient(patient_id)
        return reports, total

    async def get_all(self, skip: int = 0, limit: int = 20) -> tuple[list[Report], int]:
        reports = await self.repo.get_all(skip, limit)
        total = await self.repo.count_all()
        return reports, total
