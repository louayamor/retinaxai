from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import DoctorUser
from app.db.session import get_db
from app.oct_reports.service import OCTReportService
from app.schemas.common import MessageResponse
from app.schemas.oct_report_schema import OCTReportCreate, OCTReportRead

router = APIRouter(prefix="/oct-reports", tags=["oct_reports"])


@router.post("/", response_model=OCTReportRead, status_code=201)
async def create_oct_report(
    data: OCTReportCreate,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = OCTReportService(db)
    return await service.create(data.model_dump())


@router.get("/", response_model=list[OCTReportRead])
async def list_oct_reports(
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    service = OCTReportService(db)
    return await service.list_all(skip=skip, limit=limit)


@router.get("/patients/{patient_id}", response_model=list[OCTReportRead])
async def list_patient_oct_reports(
    patient_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = OCTReportService(db)
    return await service.get_by_patient(patient_id)


@router.get("/{report_id}", response_model=OCTReportRead)
async def get_oct_report(
    report_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = OCTReportService(db)
    return await service.get_by_id(report_id)


@router.delete("/{report_id}", response_model=MessageResponse)
async def delete_oct_report(
    report_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = OCTReportService(db)
    await service.delete(report_id)
    return MessageResponse(message="OCT report deleted successfully.")
