from __future__ import annotations
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import NotFoundException, UnprocessableEntityException
from app.models.mri_scan import MRIScan
from app.mri_scans.repository import MRIScanRepository
from app.patients.repository import PatientRepository
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


class MRIScanService:
    def __init__(self, db: AsyncSession):
        self.repo = MRIScanRepository(db)
        self.patient_repo = PatientRepository(db)
        settings.ensure_dirs()

    def _validate_file(self, file: UploadFile) -> None:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise UnprocessableEntityException(
                f"Invalid file type '{file.content_type}'. Only PNG/JPEG files are accepted."
            )

    async def _save_file(
        self, file: UploadFile, patient_id: uuid.UUID, side: str, modality: str
    ) -> str:
        if modality == "fundus":
            dest_dir = settings.fundus_dir / str(patient_id)
        else:
            dest_dir = settings.oct_dir / str(patient_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        ext = "png" if file.content_type == "image/png" else "jpg"
        dest_path = dest_dir / f"{side}.{ext}"
        content = await file.read()
        dest_path.write_bytes(content)
        return str(dest_path)

    async def upload(
        self,
        patient_id: uuid.UUID,
        left_scan: UploadFile,
        right_scan: UploadFile,
        modality: str = "fundus",
    ) -> MRIScan:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", patient_id)  # type: ignore[reportArgumentType]

        self._validate_file(left_scan)
        self._validate_file(right_scan)

        left_path = await self._save_file(left_scan, patient_id, "left", modality)
        right_path = await self._save_file(right_scan, patient_id, "right", modality)

        scan = MRIScan(
            patient_id=patient_id,
            left_scan_path=left_path,
            right_scan_path=right_path,
            modality=modality,
        )
        return await self.repo.create(scan)

    async def get_by_id(self, scan_id: uuid.UUID) -> MRIScan:
        scan = await self.repo.get_by_id(scan_id)
        if not scan:
            raise NotFoundException("MRIScan", scan_id)  # type: ignore[reportArgumentType]
        return scan

    async def get_by_patient(self, patient_id: uuid.UUID) -> list[MRIScan]:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", patient_id)  # type: ignore[reportArgumentType]
        return await self.repo.get_by_patient(patient_id)

    async def delete(self, scan_id: uuid.UUID) -> None:
        scan = await self.get_by_id(scan_id)
        left = Path(scan.left_scan_path)
        right = Path(scan.right_scan_path)
        if left.exists():
            left.unlink()
        if right.exists():
            right.unlink()
        await self.repo.delete(scan)
