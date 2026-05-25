from __future__ import annotations
import enum

from sqlalchemy import Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin


class Gender(str, enum.Enum):
    M = "M"
    F = "F"


class Patient(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "patients"

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[Gender] = mapped_column(Enum(Gender), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    medical_record_number: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    ocr_patient_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )

    mri_scans = relationship(
        "MRIScan", back_populates="patient", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="patient", cascade="all, delete-orphan"
    )
    reports = relationship(
        "Report", back_populates="patient", cascade="all, delete-orphan"
    )
    severity_reports = relationship(
        "SeverityReport", back_populates="patient", cascade="all, delete-orphan"
    )
