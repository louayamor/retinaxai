import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class BiomarkerStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VascularBiomarker(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "vascular_biomarkers"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    eye_side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    service_name: Mapped[str] = mapped_column(String(100), nullable=False)
    service_version: Mapped[str] = mapped_column(String(50), nullable=False)
    contract_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    status: Mapped[BiomarkerStatus] = mapped_column(
        Enum(BiomarkerStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=BiomarkerStatus.PENDING,
        nullable=False,
    )
    biomarkers: Mapped[dict] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    prediction = relationship("Prediction", back_populates="vascular_biomarker")
    patient = relationship("Patient")
