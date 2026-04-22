import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.prediction import Prediction


class PredictionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(
        self, prediction_id: uuid.UUID, load_patient: bool = False
    ) -> Prediction | None:
        query = select(Prediction).where(Prediction.id == prediction_id)
        if load_patient:
            query = query.options(selectinload(Prediction.patient))
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_patient(
        self,
        patient_id: uuid.UUID,
        skip: int = 0,
        limit: int = 20,
        load_patient: bool = False,
    ) -> list[Prediction]:
        query = select(Prediction).where(Prediction.patient_id == patient_id)
        if load_patient:
            query = query.options(selectinload(Prediction.patient))
        query = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_all(
        self, skip: int = 0, limit: int = 20, load_patient: bool = False
    ) -> list[Prediction]:
        query = select(Prediction)
        if load_patient:
            query = query.options(selectinload(Prediction.patient))
        query = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count_all(self) -> int:
        result = await self.db.execute(select(func.count()).select_from(Prediction))
        return result.scalar_one()

    async def count_by_patient(self, patient_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.patient_id == patient_id)
        )
        return result.scalar_one()

    async def create(self, prediction: Prediction) -> Prediction:
        self.db.add(prediction)
        await self.db.flush()
        await self.db.refresh(prediction)
        return prediction

    async def update(self, prediction: Prediction) -> Prediction:
        prediction = await self.db.merge(prediction)
        await self.db.flush()
        await self.db.refresh(prediction)
        return prediction
