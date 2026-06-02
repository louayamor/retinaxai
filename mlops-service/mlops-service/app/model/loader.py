from __future__ import annotations

from pathlib import Path
from loguru import logger


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
