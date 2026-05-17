from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from app.api.rag_schemas import RagArtifactManifest, RagManifestResponse
from app.api.rag_schemas import RagArtifactId, RagArtifactType, RagPipeline
from app.config.settings import Settings

logger = logging.getLogger(__name__)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, default=str, ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _artifact_from_json(
    artifact_id: str, path: Path, payload: Any
) -> RagArtifactManifest:
    return RagArtifactManifest(
        artifact_id=RagArtifactId(artifact_id),
        artifact_type=RagArtifactType.JSON,
        source_path=str(path),
        content_hash=_hash_payload(payload),
        content_length=len(json.dumps(payload, ensure_ascii=True, default=str)),
        content=payload,
        indexable=True,
    )


def _fetch_backend_manifest(
    backend_url: str, timeout: int = 10
) -> RagManifestResponse | None:
    url = f"{backend_url.rstrip('/')}/api/v1/rag/manifest"
    req = Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("backend_manifest_http_%d", resp.status)
                return None
            body = json.loads(resp.read().decode("utf-8"))
            return RagManifestResponse.model_validate(body)
    except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("backend_manifest_unavailable: %s", exc)
        return None


def build_rag_manifest(settings: Settings) -> RagManifestResponse:
    sources = [
        ("ocr_reports", Path(settings.ocr_output_dir) / "reports.json", "list"),
        ("clinical_metrics", settings.clinical_metrics_path, "json"),
        (
            "clinical_feature_importance",
            settings.clinical_feature_importance_path,
            "json",
        ),
        ("imaging_metrics", settings.imaging_metrics_path, "json"),
        ("clinical_features", settings.clinical_features_path, "json"),
        ("evidently_metrics", settings.evidently_metrics_path, "json"),
    ]

    artifacts: list[RagArtifactManifest] = []
    mtimes: list[float] = []

    for artifact_id, path, _kind in sources:
        if not path.is_file():
            continue
        payload = _load_json(path)
        if artifact_id == "ocr_reports" and not isinstance(payload, list):
            continue
        artifacts.append(_artifact_from_json(artifact_id, path, payload))
        mtimes.append(path.stat().st_mtime)

    meta_dir = settings.model_registry_dir / "metadata"
    if meta_dir.is_dir():
        registry_meta = {}
        for meta_file in sorted(meta_dir.glob("*.json")):
            registry_meta[meta_file.stem] = _load_json(meta_file)
        if registry_meta:
            artifacts.append(
                _artifact_from_json("model_registry_metadata", meta_dir, registry_meta)
            )
            mtimes.append(max(f.stat().st_mtime for f in meta_dir.glob("*.json")))

    backend_manifest: RagManifestResponse | None = None
    backend_url: str = getattr(settings, "backend_service_url", "")
    if backend_url:
        backend_manifest = _fetch_backend_manifest(backend_url)
    if backend_manifest and backend_manifest.artifacts:
        for art in backend_manifest.artifacts:
            try:
                validated = RagArtifactManifest.model_validate(art.model_dump())
                artifacts.append(validated)
                mtimes.append(backend_manifest.generated_at.timestamp())
            except Exception as exc:
                logger.warning("backend_artifact_skip_%s: %s", art.artifact_id, exc)

    core_ids = {
        RagArtifactId.OCR_REPORTS,
        RagArtifactId.CLINICAL_METRICS,
        RagArtifactId.CLINICAL_FEATURE_IMPORTANCE,
        RagArtifactId.IMAGING_METRICS,
    }
    present_ids = {a.artifact_id for a in artifacts}
    has_backend = any(
        a.artifact_id
        in {
            RagArtifactId.DB_PREDICTIONS,
            RagArtifactId.DB_EXPLANATIONS,
            RagArtifactId.DB_PATIENTS,
        }
        for a in artifacts
    )
    if has_backend:
        pipeline = RagPipeline.DB
    elif core_ids.issubset(present_ids):
        pipeline = RagPipeline.COMBINED
    else:
        pipeline = RagPipeline.PARTIAL

    run_id = (
        hashlib.sha1(
            "|".join(a.content_hash for a in artifacts).encode("utf-8")
        ).hexdigest()[:12]
        if artifacts
        else "none"
    )
    return RagManifestResponse(
        run_id=run_id,
        pipeline=pipeline,
        generated_at=datetime.fromtimestamp(max(mtimes), tz=UTC)
        if mtimes
        else datetime.now(tz=UTC),
        artifact_count=len(artifacts),
        artifacts=artifacts,
    )
