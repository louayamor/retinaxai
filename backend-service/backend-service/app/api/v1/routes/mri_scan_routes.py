from __future__ import annotations
import uuid
from typing import Annotated

import base64
from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import DoctorUser
from app.db.session import get_db
from app.models.mri_scan import Modality
from app.mri_scans.service import MRIScanService
from app.schemas.common import MessageResponse
from app.schemas.mri_scan_schema import MRIScanRead

router = APIRouter(tags=["mri_scans"])


@router.post("/patients/{patient_id}/scans", response_model=MRIScanRead, status_code=201)
async def upload_scans(
    patient_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    left_scan: UploadFile = File(...),
    right_scan: UploadFile = File(...),
    modality: Modality = Query(default=Modality.FUNDUS),
):
    service = MRIScanService(db)
    return await service.upload(patient_id, left_scan, right_scan, modality.value)


@router.get("/patients/{patient_id}/scans", response_model=list[MRIScanRead])
async def list_patient_scans(
    patient_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = MRIScanService(db)
    return await service.get_by_patient(patient_id)


@router.get("/scans/{scan_id}")
async def get_scan(
    scan_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = MRIScanService(db)
    scan = await service.get_by_id(scan_id)

    result = {
        "id": str(scan.id),
        "patient_id": str(scan.patient_id),
        "left_scan_path": scan.left_scan_path,
        "right_scan_path": scan.right_scan_path,
    }

    left_bytes = await service.read_scan_bytes(scan, "left")
    right_bytes = await service.read_scan_bytes(scan, "right")

    if left_bytes is not None:
        result["left_image"] = base64.b64encode(left_bytes).decode()
    if right_bytes is not None:
        result["right_image"] = base64.b64encode(right_bytes).decode()

    return result


@router.delete("/scans/{scan_id}", response_model=MessageResponse)
async def delete_scan(
    scan_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = MRIScanService(db)
    await service.delete(scan_id)
    return MessageResponse(message="MRI scan deleted successfully.")
