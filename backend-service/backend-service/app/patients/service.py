from __future__ import annotations
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.exceptions import ConflictException, NotFoundException
from app.models.patient import Patient
from app.patients.repository import PatientRepository
from app.schemas.patient_schema import PatientCreate, PatientUpdate


class PatientStats:
    def __init__(
        self,
        total: int,
        avg_age: float,
        male_count: int,
        female_count: int,
        this_month: int,
    ):
        self.total = total
        self.avg_age = round(avg_age, 1) if avg_age else 0.0
        self.male_count = male_count
        self.female_count = female_count
        self.this_month = this_month


class PatientService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = PatientRepository(db)

    async def get_stats(self) -> PatientStats:
        from datetime import datetime
        from calendar import monthrange

        now = datetime.now()
        month_start = datetime(now.year, now.month, 1)

        # Total count
        total_result = await self.db.execute(select(func.count(Patient.id)))
        total = total_result.scalar() or 0

        # Average age
        avg_result = await self.db.execute(select(func.avg(Patient.age)))
        avg_age = avg_result.scalar() or 0

        # Gender counts
        male_result = await self.db.execute(
            select(func.count(Patient.id)).where(Patient.gender == "M")
        )
        male_count = male_result.scalar() or 0

        female_result = await self.db.execute(
            select(func.count(Patient.id)).where(Patient.gender == "F")
        )
        female_count = female_result.scalar() or 0

        # This month count
        month_result = await self.db.execute(
            select(func.count(Patient.id)).where(Patient.created_at >= month_start)
        )
        this_month = month_result.scalar() or 0

        return PatientStats(total, avg_age, male_count, female_count, this_month)

    async def create(self, data: PatientCreate) -> Patient:
        existing = await self.repo.get_by_mrn(data.medical_record_number)
        if existing:
            raise ConflictException(
                f"Patient with MRN '{data.medical_record_number}' already exists."
            )
        patient = Patient(**data.model_dump())
        return await self.repo.create(patient)

    async def get_by_id(self, patient_id: uuid.UUID) -> Patient:
        patient = await self.repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", patient_id)  # type: ignore[reportArgumentType]
        return patient

    async def get_all(
        self, skip: int = 0, limit: int = 20, query: str | None = None
    ) -> tuple[list[Patient], int]:
        patients = await self.repo.get_all(skip, limit, query)
        total = await self.repo.count(query)
        return patients, total

    async def update(self, patient_id: uuid.UUID, data: PatientUpdate) -> Patient:
        patient = await self.get_by_id(patient_id)
        updates = data.model_dump(exclude_none=True)
        new_mrn = updates.get("medical_record_number")
        if new_mrn and new_mrn != patient.medical_record_number:
            existing = await self.repo.get_by_mrn(new_mrn)
            if existing:
                raise ConflictException(f"Patient with MRN '{new_mrn}' already exists.")
        for field, value in updates.items():
            setattr(patient, field, value)
        return await self.repo.update(patient)

    async def delete(self, patient_id: uuid.UUID) -> None:
        patient = await self.get_by_id(patient_id)
        await self.repo.delete(patient)
