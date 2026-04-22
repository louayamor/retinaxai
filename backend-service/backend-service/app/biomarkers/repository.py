from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.vascular_biomarker import VascularBiomarker


class BiomarkerRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_prediction_id(self, prediction_id: uuid.UUID) -> VascularBiomarker | None:
        result = await self.db.execute(
            select(VascularBiomarker).where(VascularBiomarker.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()

    async def create(self, biomarker: VascularBiomarker) -> VascularBiomarker:
        self.db.add(biomarker)
        await self.db.flush()
        await self.db.refresh(biomarker)
        return biomarker

    async def update(self, biomarker: VascularBiomarker) -> VascularBiomarker:
        await self.db.flush()
        await self.db.refresh(biomarker)
        return biomarker
