from __future__ import annotations
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.role_guard import DoctorUser
from app.db.session import get_db
from app.predictions.service import PredictionService
from app.schemas.prediction_schema import PredictionRead, PredictionRequest

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionRead, status_code=201)
async def run_prediction(
    data: PredictionRequest,
    current_user: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PredictionService(db)
    return await service.run(data, current_user.id)


@router.get("/", response_model=dict)
async def list_predictions(
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    service = PredictionService(db)
    skip = (page - 1) * size
    predictions, total = await service.get_all(skip, size)
    return {
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
        "items": [PredictionRead.model_validate(p) for p in predictions],
    }


@router.get("/patient/{patient_id}", response_model=dict)
async def list_patient_predictions(
    patient_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    service = PredictionService(db)
    skip = (page - 1) * size
    predictions, total = await service.get_by_patient(patient_id, skip, size)
    return {
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
        "items": [PredictionRead.model_validate(p) for p in predictions],
    }


@router.get("/{prediction_id}", response_model=PredictionRead)
async def get_prediction(
    prediction_id: uuid.UUID,
    _: DoctorUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = PredictionService(db)
    return await service.get_by_id(prediction_id)
