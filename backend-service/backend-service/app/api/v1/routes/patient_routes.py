from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import StaffUser
from app.db.session import get_db
from app.patients.service import PatientService, PatientStats
from app.schemas.common import MessageResponse
from app.schemas.patient_schema import PatientCreate, PatientRead, PatientUpdate

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientStatsResponse(BaseModel):
    total: int
    avg_age: float
    male_count: int
    female_count: int
    this_month: int


@router.get("/stats", response_model=PatientStatsResponse)
async def get_patient_stats(
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    stats = await service.get_stats()
    return PatientStatsResponse(
        total=stats.total,
        avg_age=stats.avg_age,
        male_count=stats.male_count,
        female_count=stats.female_count,
        this_month=stats.this_month,
    )


@router.post("/", response_model=PatientRead, status_code=201)
async def create_patient(
    data: PatientCreate,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.create(data)


@router.get("/", response_model=list[PatientRead])
async def list_patients(
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    q: str | None = Query(None, min_length=1),
):
    service = PatientService(db)
    patients, _ = await service.get_all(skip=skip, limit=limit, query=q)  # type: ignore[assignment]
    return patients


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(
    patient_id: uuid.UUID,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.get_by_id(patient_id)


@router.patch("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: uuid.UUID,
    data: PatientUpdate,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.update(patient_id, data)


@router.delete("/{patient_id}", response_model=MessageResponse)
async def delete_patient(
    patient_id: uuid.UUID,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    await service.delete(patient_id)
    return MessageResponse(message="Patient deleted successfully.")
