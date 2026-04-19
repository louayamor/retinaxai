import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.report import Report, ReportType


class ReportRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, report_id: uuid.UUID) -> Report | None:
        return await self.db.get(Report, report_id)

    async def get_by_prediction_id(self, prediction_id: uuid.UUID) -> Report | None:
        result = await self.db.execute(
            select(Report).where(Report.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()

    async def get_by_patient(
        self, patient_id: uuid.UUID, skip: int = 0, limit: int = 20
    ) -> list[Report]:
        result = await self.db.execute(
            select(Report)
            .where(Report.patient_id == patient_id)
            .order_by(Report.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_type(
        self, report_type: ReportType, skip: int = 0, limit: int = 100
    ) -> list[Report]:
        result = await self.db.execute(
            select(Report)
            .where(Report.report_type == report_type)
            .order_by(Report.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_patient_and_type(
        self, patient_id: uuid.UUID, report_type: ReportType
    ) -> list[Report]:
        result = await self.db.execute(
            select(Report)
            .where(
                Report.patient_id == patient_id,
                Report.report_type == report_type,
            )
            .order_by(Report.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_by_type(self, report_type: ReportType) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(Report)
            .where(Report.report_type == report_type)
        )
        return result.scalar_one()

    async def count_by_patient(self, patient_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(Report)
            .where(Report.patient_id == patient_id)
        )
        return result.scalar_one()

    async def create(self, report: Report) -> Report:
        self.db.add(report)
        await self.db.flush()
        await self.db.refresh(report)
        return report

    async def update(self, report: Report) -> Report:
        await self.db.flush()
        await self.db.refresh(report)
        return report
