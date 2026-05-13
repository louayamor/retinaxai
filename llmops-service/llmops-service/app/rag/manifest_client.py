from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


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
    CLINICAL_FEATURES = "clinical_features"
    EVIDENTLY_METRICS = "evidently_metrics"
    MODEL_REGISTRY_METADATA = "model_registry_metadata"


class RagArtifactType(StrEnum):
    JSON = "json"


class RagArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=RagSchemaVersion.V1)
    artifact_id: RagArtifactId
    artifact_type: RagArtifactType
    source_path: str
    content_hash: str
    content_length: int
    indexable: bool
    content: Any | None = None


class RagManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=RagSchemaVersion.V1)
    run_id: str
    pipeline: RagPipeline
    generated_at: datetime
    artifact_count: int
    artifacts: list[RagArtifact]

    @field_validator("artifacts")
    @classmethod
    def _artifacts_must_be_list(cls, value: list[RagArtifact]) -> list[RagArtifact]:
        return value


class RagManifestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=RagSchemaVersion.V1)
    run_id: str
    pipeline: RagPipeline
    generated_at: datetime
    artifact_count: int = Field(ge=0)
    artifacts: list[RagArtifact]


def fetch_manifest(url: str, timeout: float = 30.0) -> RagManifest:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    try:
        parsed = RagManifestResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid RAG manifest payload: {exc}") from exc

    if parsed.artifact_count != len(parsed.artifacts):
        raise ValueError(
            "Invalid RAG manifest payload: artifact_count does not match artifacts length"
        )

    return RagManifest.model_validate(parsed.model_dump())
