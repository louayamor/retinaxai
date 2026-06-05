from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def sync_artifacts_from_gcs(
    bucket_name: str, prefix: str, local_dir: Path
) -> None:
    """Download all artifacts under `prefix` from GCS to `local_dir`.

    Skips files that already exist and are the same size (simple cache check).
    """
    if not bucket_name or not prefix:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            logger.warning("gcs_sync_no_files", bucket=bucket_name, prefix=prefix)
            return
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = Path(blob.name).relative_to(prefix)
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == blob.size:
                continue
            blob.download_to_filename(str(dest))
            logger.info(
                "gcs_sync_downloaded",
                key=blob.name,
                dest=str(dest),
                size=blob.size,
            )
        logger.info("gcs_sync_complete", bucket=bucket_name, prefix=prefix, files=len(blobs))
    except Exception as exc:
        logger.error("gcs_sync_failed", bucket=bucket_name, prefix=prefix, error=str(exc))


def read_json_from_gcs(bucket_name: str, key: str) -> dict[str, Any] | None:
    """Read a JSON file directly from GCS.

    Returns None if the bucket is not configured, the key doesn't exist,
    or any GCS error occurs.
    """
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        if not blob.exists():
            logger.warning("gcs_json_not_found", bucket=bucket_name, key=key)
            return None
        raw = blob.download_as_bytes()
        logger.info("gcs_json_loaded", bucket=bucket_name, key=key, size=len(raw))
        return json.loads(raw)
    except Exception as exc:
        logger.error("gcs_json_read_failed", bucket=bucket_name, key=key, error=str(exc))
        return None


class GCSModelLoader:
    def __init__(self, bucket_name: str, prefix: str = "models") -> None:
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._enabled = bool(bucket_name)
        self._client = None

        if self._enabled:
            try:
                from google.cloud import storage
                self._client = storage.Client()
                logger.info("gcs_model_loader_initialized", bucket=bucket_name, prefix=prefix)
            except Exception as exc:
                logger.error("gcs_model_loader_init_failed", bucket=bucket_name, error=str(exc))
                self._enabled = False

    def download_to_cache(self, cache_dir: Path, pipeline: str, filename: str) -> Path | None:
        if not self._enabled:
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / filename
        if dest.exists():
            logger.info("model_cache_hit", pipeline=pipeline, path=str(dest))
            return dest

        key = f"{self._prefix}/{pipeline}/{filename}"
        try:
            bucket = self._client.bucket(self._bucket_name)
            blob = bucket.blob(key)
            if not blob.exists():
                logger.warning("gcs_model_not_found", key=key)
                return None
            blob.download_to_filename(str(dest))
            logger.info("gcs_model_downloaded", key=key, path=str(dest))
            return dest
        except Exception as exc:
            logger.error("gcs_model_download_failed", key=key, error=str(exc))
            return None

    def list_models(self, pipeline: str) -> list[str]:
        if not self._enabled:
            return []
        prefix = f"{self._prefix}/{pipeline}/"
        try:
            bucket = self._client.bucket(self._bucket_name)
            blobs = self._client.list_blobs(bucket, prefix=prefix)
            return [b.name[len(prefix):] for b in blobs if not b.name.endswith("/")]
        except Exception as exc:
            logger.error("gcs_model_list_failed", prefix=prefix, error=str(exc))
            return []
