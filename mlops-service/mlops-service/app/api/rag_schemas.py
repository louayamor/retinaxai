from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RagSchemaVersion(StrEnum):
    V1 = "1.0"


class RagPipeline(StrEnum):
    CLINICAL = "clinical"
    IMAGING = "imaging"
    OCR = "ocr"
    COMBINED = "combined"
    PARTIAL = "partial"


class RagArtifactId(StrEnum):
    OCR_REPORTS = "ocr_reports"
    CLINICAL_METRICS = "clinical_metrics"
    CLINICAL_FEATURE_IMPORTANCE = "clinical_feature_importance"
    IMAGING_METRICS = "imaging_metrics"


class RagArtifactType(StrEnum):
    JSON = "json"


class RagArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default=RagSchemaVersion.V1.value)
    artifact_id: RagArtifactId
    artifact_type: RagArtifactType
    source_path: str
    content_hash: str
    content_length: int = Field(ge=0)
    indexable: bool = True
    content: Any | None = None


class RagManifestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default=RagSchemaVersion.V1.value)
    run_id: str
    pipeline: RagPipeline
    generated_at: datetime
    artifact_count: int = Field(ge=0)
    artifacts: list[RagArtifactManifest]
