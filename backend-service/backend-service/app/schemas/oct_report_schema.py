from __future__ import annotations
import uuid
from datetime import datetime

from pydantic import BaseModel

from app.schemas.common import BaseResponse


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


class OCTReportRead(BaseResponse):
    id: uuid.UUID
    patient_id: uuid.UUID
    eye: str
    source_file: str
    dr_grade: str | None
    edema: bool
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
