import uuid

from sqlalchemy import Boolean, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDMixin


class OCTReport(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "oct_reports"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    eye: Mapped[str] = mapped_column(String(2), nullable=False)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    dr_grade: Mapped[str | None] = mapped_column(String(50), nullable=True)
    edema: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
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
