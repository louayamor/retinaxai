import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.report import ReportStatus, ReportType
from app.schemas.common import BaseResponse


class ReportGenerateRequest(BaseModel):
    prediction_id: uuid.UUID


class OCTReportCreate(BaseModel):
    patient_id: uuid.UUID
    eye: str
    source_file: str
    dr_grade: str | None = None
    edema: bool = False
    erm_status: str | None = None
    image_quality: float | None = None
    thickness_center_fovea: float | None = None
    thickness_average_thickness: float | None = None
    thickness_total_volume_mm3: float | None = None
    thickness_inner_superior: float | None = None
    thickness_inner_nasal: float | None = None
    thickness_inner_inferior: float | None = None
    thickness_inner_temporal: float | None = None
    thickness_outer_superior: float | None = None
    thickness_outer_nasal: float | None = None
    thickness_outer_inferior: float | None = None
    thickness_outer_temporal: float | None = None


class ReportRead(BaseResponse):
    id: uuid.UUID
    patient_id: uuid.UUID
    prediction_id: uuid.UUID | None
    generated_by: uuid.UUID
    llm_model: str
    content: str | None
    summary: str | None
    status: ReportStatus
    error_message: str | None
    report_type: ReportType
    eye: str | None
    source_file: str | None
    dr_grade: str | None
    edema: bool | None
    erm_status: str | None
    image_quality: float | None
    thickness_center_fovea: float | None
    thickness_average_thickness: float | None
    thickness_total_volume_mm3: float | None
    thickness_inner_superior: float | None
    thickness_inner_nasal: float | None
    thickness_inner_inferior: float | None
    thickness_inner_temporal: float | None
    thickness_outer_superior: float | None
    thickness_outer_nasal: float | None
    thickness_outer_inferior: float | None
    thickness_outer_temporal: float | None
    created_at: datetime
