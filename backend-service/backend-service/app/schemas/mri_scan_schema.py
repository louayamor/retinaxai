from __future__ import annotations
import uuid
from datetime import datetime

from pydantic import Field, field_validator
from app.schemas.common import BaseResponse

DATA_DIRS = [
    "/home/louay/RetinaXAI/backend-service/data",
    "/home/louay/RetinaXAI/backend-service/backend-service/data",
]


def _normalize_path(path: str) -> str:
    if not path:
        return path
    path = path.strip()
    for data_dir in DATA_DIRS:
        if path.startswith(data_dir):
            relative = path.replace(data_dir, "")
            if relative.startswith("/"):
                relative = relative[1:]
            if relative:
                return (
                    f"uploads/{relative}"
                    if not relative.startswith("uploads/")
                    else relative
                )
    if path.startswith("/") and "uploads/" in path:
        return path if path.startswith("uploads/") else path[path.index("uploads/") :]
    return path


class MRIScanRead(BaseResponse):
    id: uuid.UUID
    patient_id: uuid.UUID
    modality: str
    left_scan_path: str = Field(serialization_alias="left_scan_path")
    right_scan_path: str = Field(serialization_alias="right_scan_path")
    scanned_at: datetime
    created_at: datetime

    @field_validator("left_scan_path", "right_scan_path", mode="before")
    @classmethod
    def normalize_paths(cls, v):
        return _normalize_path(v) if isinstance(v, str) else v
