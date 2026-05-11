import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.prediction import PredictionStatus
from app.schemas.common import BaseResponse


class PredictionRequest(BaseModel):
    patient_id: uuid.UUID
    mri_scan_id: uuid.UUID
    model_name: str
    model_version: str
    input_payload: dict


class PredictionRead(BaseResponse):
    id: uuid.UUID
    patient_id: uuid.UUID
    mri_scan_id: uuid.UUID
    requested_by: uuid.UUID
    model_name: str
    model_version: str
    input_payload: dict
    output_payload: dict | None
    confidence_score: float | None
    status: PredictionStatus
    error_message: str | None
    created_at: datetime
