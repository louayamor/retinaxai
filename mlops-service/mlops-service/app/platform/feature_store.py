import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class FeatureStoreError(Exception):
    pass


class FeatureNotFoundError(FeatureStoreError):
    pass


class InMemoryFeatureStore:
    def __init__(self):
        self._features: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = {}
        self._last_access: dict[str, str] = {}

    def set(
        self, key: str, value: dict[str, Any], ttl_seconds: Optional[int] = 3600
    ) -> None:
        self._features[key] = value
        self._versions[key] = self._versions.get(key, 0) + 1
        self._last_access[key] = datetime.utcnow().isoformat()

    def get(self, key: str) -> dict[str, Any]:
        if key not in self._features:
            raise FeatureNotFoundError(f"Feature not found: {key}")
        return self._features[key]

    def delete(self, key: str) -> None:
        if key in self._features:
            del self._features[key]
        if key in self._versions:
            del self._versions[key]
        if key in self._last_access:
            del self._last_access[key]

    def exists(self, key: str) -> bool:
        return key in self._features

    def get_version(self, key: str) -> int:
        return self._versions.get(key, 0)

    def get_last_access(self, key: str) -> Optional[str]:
        return self._last_access.get(key)

    def list_keys(self) -> list[str]:
        return list(self._features.keys())

    def clear(self) -> None:
        self._features.clear()
        self._versions.clear()
        self._last_access.clear()


class RedisFeatureStore:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis
        except ImportError:
            raise FeatureStoreError("redis package not installed")

        try:
            self._client = redis.from_url(redis_url)
            self._client.ping()
            logger.info(f"Redis connected: {redis_url}")
        except Exception as e:
            raise FeatureStoreError(f"Cannot connect to Redis: {e}")

        self._prefix = "retinaxai:feature:"

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def set(
        self, key: str, value: dict[str, Any], ttl_seconds: Optional[int] = 3600
    ) -> None:
        full_key = self._make_key(key)
        self._client.set(full_key, json.dumps(value))
        version_key = f"{full_key}:version"
        self._client.incr(version_key)
        if ttl_seconds:
            self._client.expire(full_key, ttl_seconds)

    def get(self, key: str) -> dict[str, Any]:
        full_key = self._make_key(key)
        value = self._client.get(full_key)
        if value is None:
            raise FeatureNotFoundError(f"Feature not found: {key}")
        return json.loads(value)

    def delete(self, key: str) -> None:
        full_key = self._make_key(key)
        self._client.delete(full_key, f"{full_key}:version", f"{full_key}:last_access")

    def exists(self, key: str) -> bool:
        full_key = self._make_key(key)
        return self._client.exists(full_key) > 0

    def get_version(self, key: str) -> int:
        full_key = self._make_key(key)
        version = self._client.get(f"{full_key}:version")
        return int(version) if version else 0

    def get_last_access(self, key: str) -> Optional[str]:
        full_key = self._make_key(key)
        last_access = self._client.get(f"{full_key}:last_access")
        return last_access.decode() if last_access else None

    def list_keys(self) -> list[str]:
        pattern = f"{self._prefix}*"
        keys = self._client.keys(pattern)
        return [
            k.decode().replace(self._prefix, "")
            for k in keys
            if not k.decode().endswith(":version")
        ]

    def clear(self) -> None:
        pattern = f"{self._prefix}*"
        keys = self._client.keys(pattern)
        if keys:
            self._client.delete(*keys)


class FeatureStore:
    def __init__(
        self,
        use_redis: bool = True,
        redis_url: str = "redis://localhost:6379",
    ):
        if use_redis:
            try:
                self._store: InMemoryFeatureStore | RedisFeatureStore = (
                    RedisFeatureStore(redis_url)
                )
                logger.info("Using Redis feature store")
            except FeatureStoreError as e:
                logger.warning(f"Redis unavailable ({e}), using in-memory store")
                self._store = InMemoryFeatureStore()
        else:
            self._store = InMemoryFeatureStore()
            logger.info("Using in-memory feature store")

    def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl_seconds: int = 3600,
    ) -> dict[str, Any]:
        if self._store.exists(key):
            logger.debug(f"Feature cache hit: {key}")
            return self._store.get(key)

        logger.info(f"Computing feature: {key}")
        value = compute_fn()

        self._store.set(key, value, ttl_seconds)
        logger.info(f"Feature cached: {key}")

        return value

    def get(self, key: str) -> dict[str, Any]:
        return self._store.get(key)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        self._store.set(key, value, ttl_seconds)

    def delete(self, key: str) -> None:
        self._store.delete(key)

    def exists(self, key: str) -> bool:
        return self._store.exists(key)

    def get_version(self, key: str) -> int:
        return self._store.get_version(key)

    def get_last_access(self, key: str) -> Optional[str]:
        return self._store.get_last_access(key)

    def list_features(self) -> list[dict[str, Any]]:
        keys = self._store.list_keys()
        features = []
        for key in keys:
            try:
                features.append(
                    {
                        "key": key,
                        "value": json.dumps(self._store.get(key)),
                        "created_at": self._store.get_last_access(key) or datetime.utcnow().isoformat(),
                    }
                )
            except FeatureNotFoundError:
                pass
        return features

    def clear(self) -> None:
        self._store.clear()
        logger.info("Feature store cleared")


_feature_store: Optional[FeatureStore] = None


def get_feature_store(
    use_redis: bool = True,
    redis_url: str = "redis://localhost:6379",
) -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore(use_redis=use_redis, redis_url=redis_url)
    return _feature_store
