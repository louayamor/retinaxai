"""Model Registry Service for version management and promotion."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.api.schemas import ModelStage, ModelVersion
from app.services.monitoring.prometheus_metrics import (
    MODEL_PROMOTIONS_TOTAL,
    MODEL_ROLLBACKS_TOTAL,
)


class ModelRegistryError(Exception):
    """Base exception for model registry operations."""

    pass


class ModelNotFoundError(ModelRegistryError):
    """Raised when a model version is not found in registry."""

    pass


class ModelAlreadyExistsError(ModelRegistryError):
    """Raised when attempting to register an existing version."""

    pass


class ModelRegistryService:
    """Service for managing model versions, lifecycle stages, and promotions."""

    def __init__(self, registry_dir: Path):
        """Initialize model registry.

        Args:
            registry_dir: Root directory for model registry storage
        """
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.metadata_dir = self.registry_dir / "metadata"
        self.backups_dir = self.registry_dir / "backups"

        # Create directory structure
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        # Create stage directories
        for stage in ModelStage:
            (self.artifacts_dir / stage.value).mkdir(exist_ok=True)

        logger.info(f"Model registry initialized at {self.registry_dir}")

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_version_metadata_path(self, version: str) -> Path:
        """Get path to model version metadata file.

        Args:
            version: Model version string (e.g., "v1.2.0")

        Returns:
            Path to metadata JSON file
        """
        return self.metadata_dir / f"{version}.json"

    def _load_version_metadata(self, version: str) -> ModelVersion:
        """Load model version metadata from disk.

        Args:
            version: Model version string

        Returns:
            ModelVersion object

        Raises:
            ModelNotFoundError: If metadata file doesn't exist
        """
        metadata_path = self._get_version_metadata_path(version)
        if not metadata_path.exists():
            raise ModelNotFoundError(f"Model version {version} not found")

        with open(metadata_path) as f:
            data = json.load(f)

        return ModelVersion(**data)

    def _save_version_metadata(self, version: ModelVersion) -> None:
        """Save model version metadata to disk.

        Args:
            version: ModelVersion to save
        """
        metadata_path = self._get_version_metadata_path(version.version)
        with open(metadata_path, "w") as f:
            json.dump(version.model_dump(), f, indent=2, default=str)

        logger.debug(f"Saved metadata for {version.version}")

    def _copy_model_artifact(self, source_path: Path, dest_path: Path) -> str:
        """Copy model artifact and return SHA256 hash.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Returns:
            SHA256 hash of copied file
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Copy file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        # Calculate and return hash
        return self._calculate_sha256(dest_path)

    def _ensure_stage_exists(self, stage: ModelStage) -> None:
        """Ensure stage directory exists.

        Args:
            stage: Model stage enum
        """
        stage_dir = self.artifacts_dir / stage.value
        stage_dir.mkdir(exist_ok=True)

    def register_version(
        self,
        version: str,
        pipeline: str,
        source_path: Path,
        metrics: dict[str, float],
        metadata: Optional[dict] = None,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            version: Version string (e.g., "v1.2.0")
            pipeline: Pipeline type ("imaging" or "clinical")
            source_path: Path to model artifact
            metrics: Model performance metrics
            metadata: Optional additional metadata

        Returns:
            Registered ModelVersion

        Raises:
            ModelAlreadyExistsError: If version already registered
        """
        logger.info(f"Registering model version {version} for {pipeline}")

        # Check if version already exists
        try:
            self._load_version_metadata(version)
            raise ModelAlreadyExistsError(f"Model version {version} already exists")
        except ModelNotFoundError:
            pass  # Expected

        # Validate file exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Determine artifact name based on pipeline
        extension = source_path.suffix
        artifact_name = (
            f"{pipeline}_{version}_{self._calculate_sha256(source_path)[:8]}{extension}"
        )

        # Initial artifact path in staging
        stage_dir = self.artifacts_dir / ModelStage.STAGING.value
        artifact_path = stage_dir / artifact_name

        # Copy artifact and calculate hash
        file_hash = self._copy_model_artifact(source_path, artifact_path)

        # Create model version object
        model_version = ModelVersion(
            version=version,
            pipeline=pipeline,
            stage=ModelStage.STAGING,  # Always start in staging
            hash=file_hash,
            metrics=metrics,
            artifact_path=str(artifact_path.relative_to(self.registry_dir)),
            metadata=metadata or {},
            created_at=datetime.now(),
        )

        # Save metadata
        self._save_version_metadata(model_version)

        logger.info(f"Successfully registered {version} (pipeline={pipeline})")
        return model_version

    def get_version(self, version: str) -> ModelVersion:
        """Get model version details.

        Args:
            version: Version string

        Returns:
            ModelVersion object

        Raises:
            ModelNotFoundError: If version not found
        """
        return self._load_version_metadata(version)

    def list_versions(
        self, pipeline: Optional[str] = None, stage: Optional[ModelStage] = None
    ) -> list[ModelVersion]:
        """List model versions with optional filtering.

        Args:
            pipeline: Filter by pipeline (imaging/clinical)
            stage: Filter by stage

        Returns:
            List of ModelVersion objects
        """
        versions = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                version_str = metadata_file.stem
                version = self._load_version_metadata(version_str)

                # Apply filters
                if pipeline and version.pipeline != pipeline:
                    continue
                if stage and version.stage != stage:
                    continue

                versions.append(version)
            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")
                continue

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def promote_version(
        self,
        version: str,
        target_stage: ModelStage = ModelStage.PRODUCTION,
        reason: Optional[str] = None,
    ) -> ModelVersion:
        """Promote model version to target stage.

        Args:
            version: Version string to promote
            target_stage: Target stage (default: PRODUCTION)
            reason: Optional reason for promotion

        Returns:
            Updated ModelVersion

        Raises:
            ModelNotFoundError: If version not found
        """
        logger.info(f"Promoting {version} to {target_stage.value}")

        # Load version
        model_version = self._load_version_metadata(version)

        # Validate promotion path
        if model_version.stage == ModelStage.ARCHIVED:
            raise ValueError("Cannot promote archived version")

        # Create backup of current production if promoting to production
        if target_stage == ModelStage.PRODUCTION:
            self._backup_current_production(model_version.pipeline)

        # Move artifact to target stage
        artifact_filename = Path(model_version.artifact_path).name
        source_artifact = self.registry_dir / model_version.artifact_path

        dest_dir = self.artifacts_dir / target_stage.value
        dest_artifact = dest_dir / artifact_filename

        # Copy artifact to target stage
        shutil.copy2(source_artifact, dest_artifact)

        # Update metadata
        model_version.stage = target_stage
        model_version.promoted_at = datetime.now()
        model_version.artifact_path = str(dest_artifact.relative_to(self.registry_dir))

        if reason:
            if "promotion_history" not in model_version.metadata:
                model_version.metadata["promotion_history"] = []
            model_version.metadata["promotion_history"].append(
                {
                    "target_stage": target_stage.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        self._save_version_metadata(model_version)

        MODEL_PROMOTIONS_TOTAL.labels(
            pipeline=model_version.pipeline,
            target_stage=target_stage.value,
        ).inc()

        logger.info(f"Successfully promoted {version} to {target_stage.value}")
        return model_version

    def _backup_current_production(self, pipeline: str) -> None:
        """Backup current production model before promotion.

        Args:
            pipeline: Pipeline type (imaging/clinical)
        """
        current_production = self.list_versions(
            pipeline=pipeline, stage=ModelStage.PRODUCTION
        )

        if not current_production:
            logger.debug("No current production model to backup")
            return

        current_model = current_production[0]  # Should only be one production model

        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = (
            f"{current_model.pipeline}_{current_model.version}_backup_{timestamp}.bak"
        )
        backup_path = self.backups_dir / backup_filename

        # Copy current production artifact
        current_artifact = self.registry_dir / current_model.artifact_path
        if current_artifact.exists():
            shutil.copy2(current_artifact, backup_path)
            logger.info(f"Backed up production model to {backup_path}")

        # Update metadata to archived and mark as backup
        current_model.stage = ModelStage.ARCHIVED
        current_model.metadata["backup_reason"] = f"Replaced by {current_model.version}"
        self._save_version_metadata(current_model)

    def rollback_version(self, version: str, reason: str) -> ModelVersion:
        """Rollback to a previous model version.

        Args:
            version: Version string to rollback to
            reason: Reason for rollback

        Returns:
            Updated ModelVersion

        Raises:
            ModelNotFoundError: If version not found
        """
        logger.info(f"Rolling back to {version}")

        # Load target version
        target_version = self._load_version_metadata(version)

        # Archive current production
        self._backup_current_production(target_version.pipeline)

        # Promote target to production
        promoted = self.promote_version(version, ModelStage.PRODUCTION, reason)

        # Add rollback note
        promoted.metadata["rollback_from"] = reason
        self._save_version_metadata(promoted)

        MODEL_ROLLBACKS_TOTAL.labels(
            pipeline=promoted.pipeline,
            reason="rollback",
        ).inc()

        logger.info(f"Successfully rolled back to {version}")
        return promoted

    def get_current_production(self, pipeline: str) -> Optional[ModelVersion]:
        """Get current production model for a pipeline.

        Args:
            pipeline: Pipeline type (imaging/clinical)

        Returns:
            Current production ModelVersion or None
        """
        production_models = self.list_versions(
            pipeline=pipeline, stage=ModelStage.PRODUCTION
        )

        if len(production_models) == 0:
            return None
        elif len(production_models) == 1:
            return production_models[0]
        else:
            logger.warning(
                f"Found {len(production_models)} production models for {pipeline}, using latest"
            )
            return sorted(
                production_models, key=lambda v: v.promoted_at or v.created_at
            )[-1]

    def get_production_metrics(self, pipeline: str) -> dict[str, float]:
        current = self.get_current_production(pipeline)
        if current is None:
            return {}
        return current.metrics or {}

    def verify_model_integrity(self, version: str) -> bool:
        """Verify model artifact matches stored hash.

        Args:
            version: Version string to verify

        Returns:
            True if hash matches, False otherwise

        Raises:
            ModelNotFoundError: If version not found
        """
        model_version = self._load_version_metadata(version)
        artifact_path = self.registry_dir / model_version.artifact_path

        if not artifact_path.exists():
            logger.error(f"Artifact not found for {version}: {artifact_path}")
            return False

        actual_hash = self._calculate_sha256(artifact_path)
        expected_hash = model_version.hash

        is_valid = actual_hash == expected_hash
        if is_valid:
            logger.debug(f"Integrity check passed for {version}")
        else:
            logger.error(
                f"Integrity check failed for {version}: {actual_hash} != {expected_hash}"
            )

        return is_valid

    def get_promotion_history(self, version: str) -> list[dict]:
        """Get promotion history for a model version.

        Args:
            version: Version string

        Returns:
            List of promotion history entries
        """
        try:
            model_version = self._load_version_metadata(version)
            return model_version.metadata.get("promotion_history", [])
        except ModelNotFoundError:
            return []
