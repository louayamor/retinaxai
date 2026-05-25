from __future__ import annotations
import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.patient import Gender
from app.schemas.common import BaseResponse


class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    age: int
    gender: Gender
    phone: str | None = None
    address: str | None = None
    medical_record_number: str
    ocr_patient_id: str | None = None


class PatientUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    age: int | None = None
    gender: Gender | None = None
    phone: str | None = None
    address: str | None = None
    medical_record_number: str | None = None
    ocr_patient_id: str | None = None


class PatientRead(BaseResponse):
    id: uuid.UUID
    first_name: str
    last_name: str
    age: int
    gender: Gender
    phone: str | None
    address: str | None
    medical_record_number: str
    ocr_patient_id: str | None
    created_at: datetime
