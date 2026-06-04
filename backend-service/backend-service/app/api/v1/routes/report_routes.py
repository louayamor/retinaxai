from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import StaffUser
from app.db.session import get_db
from app.reports.service import ReportService
from app.schemas.report_schema import OCTReportCreate, ReportGenerateRequest, ReportRead

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/", response_model=ReportRead, status_code=201)
async def generate_report(
    data: ReportGenerateRequest,
    current_user: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = ReportService(db)
    return await service.generate(data, current_user.id)


@router.post("/oct", response_model=ReportRead, status_code=201)
async def create_oct_report(
    data: OCTReportCreate,
    current_user: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = ReportService(db)
    return await service.create_oct_report(data, current_user.id)


@router.get("/oct", response_model=dict)
async def list_oct_reports(
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    service = ReportService(db)
    reports, total = await service.get_oct_reports(skip=skip, limit=limit)
    return {
        "total": total,
        "items": [ReportRead.model_validate(r) for r in reports],
    }


@router.get("/oct/patient/{patient_id}", response_model=list[ReportRead])
async def list_patient_oct_reports(
    patient_id: uuid.UUID,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = ReportService(db)
    return await service.get_oct_report_by_patient(patient_id)


@router.get("/", response_model=dict)
async def list_reports(
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    service = ReportService(db)
    skip = (page - 1) * size
    reports, total = await service.get_all(skip, size)
    return {
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
        "items": [ReportRead.model_validate(r) for r in reports],
    }


@router.get("/patient/{patient_id}", response_model=dict)
async def list_patient_reports(
    patient_id: uuid.UUID,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    service = ReportService(db)
    skip = (page - 1) * size
    reports, total = await service.get_by_patient(patient_id, skip, size)
    return {
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
        "items": [ReportRead.model_validate(r) for r in reports],
    }


@router.get("/{report_id}", response_model=ReportRead)
async def get_report(
    report_id: uuid.UUID,
    _: StaffUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = ReportService(db)
    return await service.get_by_id(report_id)
