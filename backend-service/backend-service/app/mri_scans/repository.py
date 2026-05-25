from __future__ import annotations
import uuid

from app.models.mri_scan import MRIScan
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class MRIScanRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, scan_id: uuid.UUID) -> MRIScan | None:
        return await self.db.get(MRIScan, scan_id)

    async def get_by_patient(self, patient_id: uuid.UUID) -> list[MRIScan]:
        result = await self.db.execute(
            select(MRIScan).where(MRIScan.patient_id == patient_id)
        )
        return list(result.scalars().all())

    async def create(self, scan: MRIScan) -> MRIScan:
        self.db.add(scan)
        await self.db.flush()
        await self.db.refresh(scan)
        return scan

    async def delete(self, scan: MRIScan) -> None:
        await self.db.delete(scan)
        await self.db.flush()
