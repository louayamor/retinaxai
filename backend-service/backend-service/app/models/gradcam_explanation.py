from __future__ import annotations
import uuid

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class GradCAMExplanation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "gradcam_explanations"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    left_eye_explanation: Mapped[str] = mapped_column(String(1000), nullable=False)
    right_eye_explanation: Mapped[str] = mapped_column(String(1000), nullable=False)
    highlighted_regions: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    prediction = relationship("Prediction", back_populates="gradcam_explanation")
