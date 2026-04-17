import enum
import uuid

from sqlalchemy import JSON, Enum, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class PredictionStatus(str, enum.Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"  # Prediction succeeded but XAI failed
    FAILED = "FAILED"


class Prediction(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_patient_created", "patient_id", "created_at"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    mri_scan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mri_scans.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    input_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    output_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus),
        default=PredictionStatus.PENDING,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    patient = relationship("Patient", back_populates="predictions")
    mri_scan = relationship("MRIScan")
    report = relationship("Report", back_populates="prediction", uselist=False)
    explanation = relationship(
        "PredictionExplanation", back_populates="prediction", uselist=False
    )
    gradcam_explanation = relationship(
        "GradCAMExplanation", back_populates="prediction", uselist=False
    )
    severity_report = relationship(
        "SeverityReport", back_populates="prediction", uselist=False
    )
