from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.db.session import get_db
from app.patients.service import PatientService
from app.schemas.common import MessageResponse
from app.schemas.patient_schema import PatientCreate, PatientRead, PatientUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=PatientRead, status_code=201)
async def create_patient(
    data: PatientCreate,
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.create(data)


@router.get("/", response_model=list[PatientRead])
async def list_patients(
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    service = PatientService(db)
    patients, _ = await service.get_all(skip=skip, limit=limit)  # type: ignore[assignment]
    return patients


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(
    patient_id: uuid.UUID,
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.get_by_id(patient_id)


@router.patch("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: uuid.UUID,
    data: PatientUpdate,
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    return await service.update(patient_id, data)


@router.delete("/{patient_id}", response_model=MessageResponse)
async def delete_patient(
    patient_id: uuid.UUID,
    _: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PatientService(db)
    await service.delete(patient_id)
    return MessageResponse(message="Patient deleted successfully.")
