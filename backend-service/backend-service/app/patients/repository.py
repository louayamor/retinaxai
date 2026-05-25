from __future__ import annotations
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient


class PatientRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, patient_id: uuid.UUID) -> Patient | None:
        return await self.db.get(Patient, patient_id)

    async def get_by_mrn(self, mrn: str) -> Patient | None:
        result = await self.db.execute(
            select(Patient).where(Patient.medical_record_number == mrn)
        )
        return result.scalar_one_or_none()

    async def get_all(self, skip: int = 0, limit: int = 20, query: str | None = None) -> list[Patient]:
        stmt = select(Patient)
        if query:
            q = f"%{query.strip()}%"
            stmt = stmt.where(
                or_(
                    Patient.first_name.ilike(q),
                    Patient.last_name.ilike(q),
                    Patient.medical_record_number.ilike(q),
                    Patient.ocr_patient_id.ilike(q),
                )
            )
        stmt = stmt.order_by(Patient.created_at.desc(), Patient.id.desc())
        stmt = stmt.offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count(self, query: str | None = None) -> int:
        stmt = select(func.count()).select_from(Patient)
        if query:
            q = f"%{query.strip()}%"
            stmt = stmt.where(
                or_(
                    Patient.first_name.ilike(q),
                    Patient.last_name.ilike(q),
                    Patient.medical_record_number.ilike(q),
                    Patient.ocr_patient_id.ilike(q),
                )
            )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    async def create(self, patient: Patient) -> Patient:
        self.db.add(patient)
        await self.db.flush()
        await self.db.refresh(patient)
        return patient

    async def update(self, patient: Patient) -> Patient:
        await self.db.flush()
        await self.db.refresh(patient)
        return patient

    async def delete(self, patient: Patient) -> None:
        await self.db.delete(patient)
        await self.db.flush()
