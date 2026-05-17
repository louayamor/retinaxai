from typing import Any, Optional

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from app.core.exceptions import NotFoundException
from app.platform.feature_store import FeatureNotFoundError, get_feature_store


class FeatureSetRequest(BaseModel):
    key: str
    value: dict[str, Any]
    ttl_seconds: Optional[int] = 3600


class FeatureGetResponse(BaseModel):
    key: str
    value: dict[str, Any]
    version: int


class FeatureListResponse(BaseModel):
    features: list[dict[str, Any]]
    total: int


router = APIRouter(prefix="/features", tags=["features"])


@router.post("/set")
async def set_feature(
    request: FeatureSetRequest,
) -> dict[str, str]:
    store = get_feature_store()
    store.set(request.key, request.value, request.ttl_seconds)
    return {"status": "ok", "key": request.key}


@router.get("/get/{key}", response_model=FeatureGetResponse)
async def get_feature(
    key: str,
) -> FeatureGetResponse:
    store = get_feature_store()
    try:
        value = store.get(key)
        version = store.get_version(key)
        return FeatureGetResponse(key=key, value=value, version=version)
    except FeatureNotFoundError:
        raise NotFoundException("Feature", key)


@router.delete("/delete/{key}")
async def delete_feature(
    key: str,
) -> dict[str, str]:
    store = get_feature_store()
    store.delete(key)
    return {"status": "deleted", "key": key}


@router.get("/exists/{key}")
async def feature_exists(
    key: str,
) -> dict[str, Any]:
    store = get_feature_store()
    return {"key": key, "exists": store.exists(key)}


@router.get("/list", response_model=FeatureListResponse)
async def list_features(
    store=Depends(get_feature_store),
) -> FeatureListResponse:
    features = store.list_features()
    return FeatureListResponse(features=features, total=len(features))


@router.post("/clear")
async def clear_features(
    store=Depends(get_feature_store),
) -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}
