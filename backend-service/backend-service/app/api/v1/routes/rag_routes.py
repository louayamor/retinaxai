from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.patient import Patient
from app.models.prediction import Prediction
from app.models.prediction_explanation import PredictionExplanation

router = APIRouter(prefix="/rag", tags=["rag"])


class RagArtifactId(StrEnum):
    DB_PREDICTIONS = "db_predictions"
    DB_EXPLANATIONS = "db_explanations"
    DB_PATIENTS = "db_patients"


class RagArtifactType(StrEnum):
    JSON = "json"


class RagPipeline(StrEnum):
    DB = "db"


class RagArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default="1.0")
    artifact_id: RagArtifactId
    artifact_type: RagArtifactType
    source_path: str = "postgresql://backend"
    content_hash: str
    content_length: int = Field(ge=0)
    indexable: bool = True
    content: Any = None


class RagManifestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default="1.0")
    run_id: str
    pipeline: RagPipeline
    generated_at: datetime
    artifact_count: int = Field(ge=0)
    artifacts: list[RagArtifactManifest]


def _hash(content: Any) -> str:
    raw = json.dumps(content, sort_keys=True, default=str, ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _build_artifact(
    artifact_id: RagArtifactId,
    content: list[dict],
) -> RagArtifactManifest:
    return RagArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=RagArtifactType.JSON,
        content_hash=_hash(content),
        content_length=len(json.dumps(content, ensure_ascii=True, default=str)),
        content=content,
        indexable=True,
    )


@router.get("/manifest", response_model=RagManifestResponse)
async def get_rag_manifest(
    db: AsyncSession = Depends(get_db),
) -> RagManifestResponse:
    artifacts: list[RagArtifactManifest] = []
    mtimes: list[float] = []

    now = datetime.now(timezone.utc)

    # --- Predictions ---
    pred_stmt = (
        select(Prediction)
        .options(selectinload(Prediction.patient))
        .order_by(Prediction.created_at.desc())
        .limit(500)
    )
    pred_result = await db.execute(pred_stmt)
    predictions = pred_result.scalars().all()

    pred_payload: list[dict] = []
    for p in predictions:
        entry = {
            "id": str(p.id),
            "patient_id": str(p.patient_id),
            "patient": {
                "age": p.patient.age if p.patient else None,
                "gender": p.patient.gender.value
                if p.patient and p.patient.gender
                else None,
            }
            if p.patient
            else None,
            "model_name": p.model_name,
            "model_version": p.model_version,
            "status": p.status.value if p.status else None,
            "confidence_score": p.confidence_score,
            "output_payload": p.output_payload,
            "error_message": p.error_message,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        pred_payload.append(entry)
    if pred_payload:
        artifacts.append(_build_artifact(RagArtifactId.DB_PREDICTIONS, pred_payload))
        mtimes.append(
            predictions[0].created_at.timestamp()
            if predictions[0].created_at
            else now.timestamp()
        )

    # --- XAI Explanations ---
    exp_stmt = (
        select(PredictionExplanation)
        .options(selectinload(PredictionExplanation.prediction))
        .order_by(PredictionExplanation.created_at.desc())
        .limit(500)
    )
    exp_result = await db.execute(exp_stmt)
    explanations = exp_result.scalars().all()

    exp_payload: list[dict] = []
    for e in explanations:
        entry = {
            "id": str(e.id),
            "prediction_id": str(e.prediction_id),
            "model_used": e.model_used,
            "status": e.status.value if e.status else None,
            "summary": e.summary,
            "content": e.content,
            "shap_values": e.shap_values,
            "error_message": e.error_message,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        exp_payload.append(entry)
    if exp_payload:
        artifacts.append(_build_artifact(RagArtifactId.DB_EXPLANATIONS, exp_payload))
        mtimes.append(
            explanations[0].created_at.timestamp()
            if explanations[0].created_at
            else now.timestamp()
        )

    # --- Patients ---
    pat_stmt = select(Patient).order_by(Patient.created_at.desc()).limit(500)
    pat_result = await db.execute(pat_stmt)
    patients = pat_result.scalars().all()

    pat_payload: list[dict] = []
    for pat in patients:
        entry = {
            "id": str(pat.id),
            "age": pat.age,
            "gender": pat.gender.value if pat.gender else None,
            "medical_record_number": pat.medical_record_number,
            "ocr_patient_id": pat.ocr_patient_id,
            "created_at": pat.created_at.isoformat() if pat.created_at else None,
        }
        pat_payload.append(entry)
    if pat_payload:
        artifacts.append(_build_artifact(RagArtifactId.DB_PATIENTS, pat_payload))
        mtimes.append(
            patients[0].created_at.timestamp()
            if patients[0].created_at
            else now.timestamp()
        )

    run_id = (
        hashlib.sha1("|".join(a.content_hash for a in artifacts).encode()).hexdigest()[
            :12
        ]
        if artifacts
        else "none"
    )

    return RagManifestResponse(
        run_id=run_id,
        pipeline=RagPipeline.DB,
        generated_at=datetime.fromtimestamp(max(mtimes), tz=timezone.utc)
        if mtimes
        else now,
        artifact_count=len(artifacts),
        artifacts=artifacts,
    )
