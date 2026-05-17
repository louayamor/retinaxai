from http import HTTPStatus
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.dependencies import get_settings
from app.api.schemas import (
    ModelDetailResponse,
    ModelListResponse,
    ModelPromotionRequest,
    ModelPromotionResponse,
    ModelRegisterResponse,
    ModelRollbackRequest,
    ModelStage,
    CurrentProductionResponse,
)
from app.config.settings import Settings
from app.core.exceptions import (
    ConflictException,
    MLOpsException,
    NotFoundException,
    UnprocessableEntityException,
)
from app.platform.event_client import send_raw_event
from app.registry.model_registry import (
    ModelRegistryError,
    ModelNotFoundError,
    ModelRegistryService,
)

router = APIRouter(prefix="/models", tags=["models"])


def get_registry_service(
    settings: Settings = Depends(get_settings),
) -> ModelRegistryService:
    """Get model registry service instance."""
    registry_dir = settings.model_registry_dir
    return ModelRegistryService(registry_dir)


def _validate_pipeline(pipeline: str) -> None:
    if pipeline not in ["imaging"]:
        raise UnprocessableEntityException("pipeline must be 'imaging'")


def _handle_model_not_found(version: str) -> None:
    """Convert ModelNotFoundError to NotFoundException. Call from except block."""
    raise NotFoundException("Model version", version)


def _handle_unexpected(context: str, exc: Exception) -> None:
    logger.error(f"{context} failed: {exc}")
    raise MLOpsException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        detail=f"{context} failed: {str(exc)}",
        error_code=f"{context.upper().replace(' ', '_')}_ERROR",
    )


@router.post(
    "/register",
    response_model=ModelRegisterResponse,
    status_code=HTTPStatus.CREATED,
    summary="Register a new model version",
)
async def register_model_version(
    version: str,
    pipeline: str,
    source_path: str,
    metrics: dict,
    metadata: Optional[dict] = None,
    settings: Settings = Depends(get_settings),
    service: ModelRegistryService = Depends(get_registry_service),
) -> ModelRegisterResponse:
    """Register a new model version in the registry."""
    logger.info(f"Registering model {version} for pipeline {pipeline}")

    try:
        _validate_pipeline(pipeline)

        source_path_obj = Path(source_path)
        if not source_path_obj.exists():
            raise UnprocessableEntityException(
                f"Source path does not exist: {source_path}"
            )

        model_version = service.register_version(
            version=version,
            pipeline=pipeline,
            source_path=source_path_obj,
            metrics=metrics,
            metadata=metadata or {},
        )

        logger.info(f"Successfully registered model {version}")

        await send_raw_event(
            event="model.registered",
            data={
                "version": version,
                "pipeline": pipeline,
            },
            room="models",
        )

        return ModelRegisterResponse(
            model=model_version,
            message=f"Model {version} registered successfully in staging",
            next_action=f"Run GET /models/{version} to view details, then POST /models/{version}/promote to deploy to production",
        )

    except ModelRegistryError as e:
        logger.error(f"Registry error: {e}")
        raise ConflictException(str(e))
    except UnprocessableEntityException:
        raise
    except Exception as e:
        _handle_unexpected("Registration", e)


@router.post(
    "/{version}/promote",
    response_model=ModelPromotionResponse,
    summary="Promote a model version to production",
)
async def promote_model(
    version: str,
    reason: Optional[str] = None,
    service: ModelRegistryService = Depends(get_registry_service),
) -> ModelPromotionResponse:
    """Promote a model version to production."""
    logger.info(f"Promoting model {version} to production")

    try:
        model_version = service.get_version(version)
        current_production = service.get_current_production(model_version.pipeline)

        previous_version = current_production.version if current_production else None

        promoted = service.promote_version(
            version=version, target_stage=ModelStage.PRODUCTION, reason=reason
        )

        logger.info(f"Successfully promoted {version} to production")

        await send_raw_event(
            event="model.promoted",
            data={
                "version": version,
                "previous_version": previous_version,
            },
            room="models",
        )

        return ModelPromotionResponse(
            success=True,
            previous_version=previous_version,
            new_version=promoted.version,
            promotion_time=promoted.promoted_at,
            notes=f"Model {version} is now in production. Previous version has been archived.",
        )

    except ModelNotFoundError:
        _handle_model_not_found(version)
    except Exception as e:
        _handle_unexpected("Promotion", e)


@router.post(
    "/{version}/rollback",
    response_model=ModelPromotionResponse,
    summary="Rollback to a previous model version",
)
async def rollback_model(
    version: str,
    request: ModelRollbackRequest,
    service: ModelRegistryService = Depends(get_registry_service),
) -> ModelPromotionResponse:
    """Rollback to a previous model version."""
    logger.info(f"Rolling back to model {version}")

    try:
        rollout_version = service.get_version(version)
        current_production = service.get_current_production(rollout_version.pipeline)

        current_version = current_production.version if current_production else None

        rolled_back = service.rollback_version(version, request.reason)

        logger.info(f"Successfully rolled back to {version}")

        await send_raw_event(
            event="model.rolled_back",
            data={
                "version": version,
                "previous_version": current_version,
                "reason": request.reason,
            },
            room="models",
        )

        return ModelPromotionResponse(
            success=True,
            previous_version=current_version,
            new_version=rolled_back.version,
            promotion_time=rolled_back.promoted_at,
            notes=f"Rolled back from {current_version} to {version}. Reason: {request.reason}",
        )

    except ModelNotFoundError:
        _handle_model_not_found(version)
    except Exception as e:
        _handle_unexpected("Rollback", e)


@router.get("/", response_model=ModelListResponse, summary="List all model versions")
async def list_models(
    pipeline: Optional[str] = None,
    stage: Optional[ModelStage] = None,
    service: ModelRegistryService = Depends(get_registry_service),
) -> ModelListResponse:
    """List all model versions with optional filtering."""
    try:
        if pipeline:
            _validate_pipeline(pipeline)

        models = service.list_versions(pipeline=pipeline, stage=stage)

        return ModelListResponse(
            models=models,
            total=len(models),
            staging_count=len([m for m in models if m.stage == ModelStage.STAGING]),
            production_count=len(
                [m for m in models if m.stage == ModelStage.PRODUCTION]
            ),
            archived_count=len([m for m in models if m.stage == ModelStage.ARCHIVED]),
        )

    except Exception as e:
        _handle_unexpected("Failed to list models", e)


@router.get(
    "/{version}",
    response_model=ModelDetailResponse,
    summary="Get details for a specific model version",
)
async def get_model(
    version: str,
    service: ModelRegistryService = Depends(get_registry_service),
) -> ModelDetailResponse:
    """Get detailed information for a specific model version."""
    try:
        model = service.get_version(version)
        current_production = (
            service.get_current_production(model.pipeline) if model.pipeline else None
        )

        promotion_history = service.get_promotion_history(version)

        return ModelDetailResponse(
            model=model,
            is_current_production=(
                current_production is not None and current_production.version == version
            ),
            can_promote=(model.stage == ModelStage.STAGING),
            can_rollback=(model.stage == ModelStage.PRODUCTION),
            promotion_history=promotion_history,
        )

    except ModelNotFoundError:
        _handle_model_not_found(version)
    except Exception as e:
        _handle_unexpected("Failed to get model", e)


@router.get(
    "/production/current",
    response_model=CurrentProductionResponse,
    summary="Get current production models",
)
async def get_current_production(
    service: ModelRegistryService = Depends(get_registry_service),
) -> CurrentProductionResponse:
    """Get the currently deployed production models for each pipeline."""
    try:
        imaging_model = service.get_current_production("imaging")

        return CurrentProductionResponse(
            imaging=imaging_model,
            promoted_at=(imaging_model.promoted_at if imaging_model else None),
        )

    except Exception as e:
        _handle_unexpected("Failed to get production models", e)


@router.post(
    "/{version}/stage", summary="Move model to staging (default initial stage)"
)
async def stage_model(
    version: str,
    service: ModelRegistryService = Depends(get_registry_service),
) -> JSONResponse:
    """Re-stage an archived or production model back to staging."""
    try:
        service.promote_version(
            version, ModelStage.STAGING, reason="Re-staging for testing"
        )

        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "success": True,
                "version": version,
                "stage": ModelStage.STAGING.value,
                "message": f"Model {version} moved to staging",
            },
        )

    except ModelNotFoundError:
        _handle_model_not_found(version)
    except Exception as e:
        _handle_unexpected("Staging", e)
