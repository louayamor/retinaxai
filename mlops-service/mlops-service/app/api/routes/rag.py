from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.config.settings import Settings
from app.api.rag_schemas import RagManifestResponse
from app.inference.rag_manifest_service import build_rag_manifest


router = APIRouter()


@router.get("/rag/manifest", response_model=RagManifestResponse)
def get_rag_manifest(settings: Settings = Depends(get_settings)) -> RagManifestResponse:
    return build_rag_manifest(settings)
