from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.biomarkers.repository import BiomarkerRepository
from app.models.vascular_biomarker import VascularBiomarker


class BiomarkerService:
    def __init__(self, db: AsyncSession):
        self.repo = BiomarkerRepository(db)

    async def get_by_prediction_id(self, prediction_id: uuid.UUID) -> VascularBiomarker | None:
        return await self.repo.get_by_prediction_id(prediction_id)

    async def create(self, biomarker: VascularBiomarker) -> VascularBiomarker:
        return await self.repo.create(biomarker)

    async def update(self, biomarker: VascularBiomarker) -> VascularBiomarker:
        return await self.repo.update(biomarker)
