from __future__ import annotations
import uuid

from app.models.oct_report import OCTReport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class OCTReportRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, data: dict) -> OCTReport:
        report = OCTReport(**data)
        self._session.add(report)
        await self._session.flush()
        await self._session.refresh(report)
        return report

    async def get_by_id(self, report_id: uuid.UUID) -> OCTReport | None:
        result = await self._session.execute(
            select(OCTReport).where(OCTReport.id == report_id)
        )
        return result.scalar_one_or_none()

    async def get_by_patient(self, patient_id: uuid.UUID) -> list[OCTReport]:
        result = await self._session.execute(
            select(OCTReport)
            .where(OCTReport.patient_id == patient_id)
            .order_by(OCTReport.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[OCTReport]:
        result = await self._session.execute(
            select(OCTReport)
            .order_by(OCTReport.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, report: OCTReport) -> None:
        await self._session.delete(report)
        await self._session.flush()
