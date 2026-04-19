import enum
import uuid

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class ReportStatus(str, enum.Enum):
    GENERATING = "GENERATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ReportType(str, enum.Enum):
    LLM = "LLM"
    OCT = "OCT"


class Report(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "reports"
    __table_args__ = (Index("ix_reports_status", "status"),)

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=True,
    )
    generated_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus),
        default=ReportStatus.GENERATING,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    report_type: Mapped[ReportType] = mapped_column(
        Enum(ReportType),
        default=ReportType.LLM,
        nullable=False,
    )

    eye: Mapped[str | None] = mapped_column(String(2), nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dr_grade: Mapped[str | None] = mapped_column(String(50), nullable=True)
    edema: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    erm_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    image_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_center_fovea: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_average_thickness: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_total_volume_mm3: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_inner_superior: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_inner_nasal: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_inner_inferior: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_inner_temporal: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_outer_superior: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_outer_nasal: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_outer_inferior: Mapped[float | None] = mapped_column(Float, nullable=True)
    thickness_outer_temporal: Mapped[float | None] = mapped_column(Float, nullable=True)

    patient = relationship("Patient", back_populates="reports")
    prediction = relationship("Prediction", back_populates="report")
