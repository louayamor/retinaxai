import enum
import uuid

from sqlalchemy import JSON, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class ExplanationStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionExplanation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "prediction_explanations"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[ExplanationStatus] = mapped_column(
        Enum(ExplanationStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=ExplanationStatus.PENDING,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    prediction = relationship("Prediction", back_populates="explanation")
