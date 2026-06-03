from __future__ import annotations

from loguru import logger


class GCSStorageService:
    def __init__(self, bucket_name: str) -> None:
        self._bucket_name = bucket_name
        self._enabled = bool(bucket_name)
        self._client = None

        if self._enabled:
            try:
                from google.cloud import storage

                self._client = storage.Client()
                logger.bind(bucket=bucket_name).info("gcs_storage_initialized")
            except Exception as exc:
                logger.bind(bucket=bucket_name, error=str(exc)).error("gcs_storage_init_failed")
                self._enabled = False

    def get_uri(self, key: str) -> str:
        return f"gs://{self._bucket_name}/{key}"

    @staticmethod
    def parse_gcs_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("gs://"):
            raise ValueError(f"Not a GCS URI: {uri}")
        parts = uri[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    async def upload(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        if not self._enabled:
            raise RuntimeError("GCS not configured")
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(key)
        blob.upload_from_string(data, content_type=content_type)
        logger.bind(key=key, size=len(data)).info("gcs_upload_complete")
        return self.get_uri(key)

    async def download(self, key: str) -> bytes | None:
        if not self._enabled:
            return None
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    async def delete(self, key: str) -> bool:
        if not self._enabled:
            return False
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(key)
        if blob.exists():
            blob.delete()
            logger.bind(key=key).info("gcs_delete_complete")
            return True
        return False
