from __future__ import annotations
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import NotFoundException, UnprocessableEntityException
from app.core.storage import GCSStorageService
from app.models.mri_scan import MRIScan
from app.mri_scans.repository import MRIScanRepository
from app.patients.repository import PatientRepository
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


class MRIScanService:
    def __init__(self, db: AsyncSession) -> None:
        self.repo = MRIScanRepository(db)
        self.patient_repo = PatientRepository(db)
        self._gcs = GCSStorageService(settings.GCS_BUCKET_UPLOADS)
        settings.ensure_dirs()

    def _validate_file(self, file: UploadFile) -> None:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise UnprocessableEntityException(
                f"Invalid file type '{file.content_type}'. Only PNG/JPEG files are accepted."
            )

    async def _save_file(
        self, file: UploadFile, patient_id: uuid.UUID, side: str, modality: str
    ) -> str:
        ext = "png" if file.content_type == "image/png" else "jpg"
        key = f"patients/{patient_id}/{modality}/{side}.{ext}"
        content = await file.read()

        if self._gcs._enabled:
            return await self._gcs.upload(key, content, content_type=file.content_type)

        if modality == "fundus":
            dest_dir = settings.fundus_dir / str(patient_id)
        else:
            dest_dir = settings.oct_dir / str(patient_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{side}.{ext}"
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
            raise NotFoundException("Patient", patient_id)

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
            raise NotFoundException("MRIScan", scan_id)
        return scan

    async def get_by_patient(self, patient_id: uuid.UUID) -> list[MRIScan]:
        patient = await self.patient_repo.get_by_id(patient_id)
        if not patient:
            raise NotFoundException("Patient", patient_id)
        return await self.repo.get_by_patient(patient_id)

    async def delete(self, scan_id: uuid.UUID) -> None:
        scan = await self.get_by_id(scan_id)

        if self._gcs._enabled:
            for path in (scan.left_scan_path, scan.right_scan_path):
                if path.startswith("gs://"):
                    _, key = GCSStorageService.parse_gcs_uri(path)
                    await self._gcs.delete(key)
        else:
            left = Path(scan.left_scan_path)
            right = Path(scan.right_scan_path)
            if left.exists():
                left.unlink()
            if right.exists():
                right.unlink()

        await self.repo.delete(scan)

    async def read_scan_bytes(self, scan: MRIScan, side: str) -> bytes | None:
        path = scan.left_scan_path if side == "left" else scan.right_scan_path
        if not path:
            return None

        if path.startswith("gs://"):
            _, key = GCSStorageService.parse_gcs_uri(path)
            return await self._gcs.download(key)

        local = Path(path)
        if local.exists():
            return local.read_bytes()
        return None
