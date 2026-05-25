from __future__ import annotations
import uuid

from app.core.exceptions import NotFoundException
from app.models.oct_report import OCTReport
from app.oct_reports.repository import OCTReportRepository
from app.patients.repository import PatientRepository
from sqlalchemy.ext.asyncio import AsyncSession


class OCTReportService:
    def __init__(self, db: AsyncSession):
        self.repo = OCTReportRepository(db)
        self.patient_repo = PatientRepository(db)

    async def create(self, data: dict) -> OCTReport:
        patient = await self.patient_repo.get_by_id(data["patient_id"])
        if not patient:
            raise NotFoundException("Patient", str(data["patient_id"]))
        return await self.repo.create(data)

    async def get_by_id(self, report_id: uuid.UUID) -> OCTReport:
        report = await self.repo.get_by_id(report_id)
        if not report:
            raise NotFoundException("OCTReport", str(report_id))
        return report

    async def get_by_patient(self, patient_id: uuid.UUID) -> list[OCTReport]:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", str(patient_id))
        return await self.repo.get_by_patient(patient_id)

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[OCTReport]:
        return await self.repo.list_all(skip=skip, limit=limit)

    async def delete(self, report_id: uuid.UUID) -> None:
        report = await self.get_by_id(report_id)
        await self.repo.delete(report)
