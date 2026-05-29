from loguru import logger
from pathlib import Path
from app.constants import PARAMS_FILE_PATH
from app.utils.common import read_yaml
from app.training.pipeline.stage_01_data_ingestion import run as imaging_ingest
from app.training.pipeline.stage_02_data_cleaning import run as imaging_clean
from app.training.pipeline.stage_03_data_transformation import (
    run as imaging_transform,
)
from app.training.pipeline.stage_04_model_trainer import run as imaging_train
from app.training.pipeline.stage_05_model_evaluation import (
    run as imaging_evaluate,
)
from app.registry.model_registry import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelRegistryService,
)
from app.config.settings import settings


class TrainingPipeline:
    def __init__(self):
        self.registry_service = ModelRegistryService(
            settings.artifacts_root / "model_registry"
        )

    def _generate_version(self, pipeline: str) -> str:
        """Generate version string automatically (e.g., v1.2.3)."""
        existing = self.registry_service.list_versions(pipeline=pipeline)
        if not existing:
            candidate = "v1.0.0"
        else:
            latest = sorted(existing, key=lambda v: v.created_at)[-1]
            major, minor, _ = latest.version.lstrip("v").split(".")
            candidate = f"v{major}.{int(minor) + 1}.0"

        for _ in range(100):
            try:
                self.registry_service.get_version(candidate)
            except ModelNotFoundError:
                return candidate
            major, minor, patch = candidate.lstrip("v").split(".")
            candidate = f"v{major}.{int(minor) + 1}.0"

        raise RuntimeError(
            f"Cannot generate unique version for {pipeline}"
        )

    def _register_model(
        self, pipeline: str, version: str, model_path: Path, metrics: dict
    ):
        """Register trained model in model registry."""
        try:
            # Check if model file exists
            if not model_path.exists():
                logger.warning(f"Model file not found for registration: {model_path}")
                return

            # Extract key metrics based on pipeline
            model_metrics = {}
            if pipeline == "imaging":
                model_metrics = {
                    "accuracy": metrics.get("eyepacs_test", {}).get("accuracy", 0),
                    "quadratic_weighted_kappa": metrics.get("eyepacs_test", {}).get(
                        "quadratic_weighted_kappa", 0
                    ),
                    "roc_auc_macro": metrics.get("eyepacs_test", {}).get(
                        "roc_auc_macro", 0
                    ),
                    "macro_f1": metrics.get("eyepacs_test", {}).get("macro_f1", 0),
                }
            elif pipeline == "lesion":
                model_metrics = {
                    "dice_mean": metrics.get("dice_mean", 0),
                    "dice_ma": metrics.get("per_class_dice", {}).get("ma", 0),
                    "dice_he": metrics.get("per_class_dice", {}).get("he", 0),
                    "dice_ex": metrics.get("per_class_dice", {}).get("ex", 0),
                    "dice_se": metrics.get("per_class_dice", {}).get("se", 0),
                }

            # Generate version if not provided
            if not version:
                version = self._generate_version(pipeline)

            # Register model
            model_metadata = {
                "training_timestamp": metrics.get("timestamp", "N/A"),
                "test_samples": metrics.get("num_samples", 0),
                "git_commit": "N/A",  # Can be populated from env
            }

            self.registry_service.register_version(
                version=version,
                pipeline=pipeline,
                source_path=model_path,
                metrics=model_metrics,
                metadata=model_metadata,
            )
            logger.info(f"Model {pipeline} v{version} registered successfully")

        except (ModelAlreadyExistsError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to register {pipeline} model: {e}")
            # Non-critical error - don't fail training if registration fails

    def run_lesion(self) -> dict:
        """Run lesion detection training pipeline."""
        logger.info("=== lesion pipeline started ===")

        from app.training.pipeline.lesion import (
            stage_01_ddr_ingestion,
            stage_02_ddr_transformation,
            stage_03_lesion_training,
            stage_04_lesion_evaluation,
        )

        stage_01_ddr_ingestion.run()
        stage_02_ddr_transformation.run()
        stage_03_lesion_training.run()

        metrics = stage_04_lesion_evaluation.run()

        model_path = settings.lesion_model_path
        version = self._generate_version("lesion")
        self._register_model(
            pipeline="lesion",
            version=version,
            model_path=model_path,
            metrics=metrics,
        )

        logger.info("=== lesion pipeline complete ===")
        return {"pipeline": "lesion", "metrics": metrics, "version": version}

    def run_imaging(self) -> dict:
        """Run imaging training pipeline."""
        logger.info("=== imaging pipeline started ===")

        imaging_ingest()
        imaging_clean()
        imaging_transform()
        imaging_train()

        # Get metrics from evaluation
        metrics = imaging_evaluate()

        # Register the trained model
        model_path = settings.imaging_model_path
        version = self._generate_version("imaging")
        self._register_model(
            pipeline="imaging", version=version, model_path=model_path, metrics=metrics
        )

        logger.info("=== imaging pipeline complete ===")
        return {"pipeline": "imaging", "metrics": metrics, "version": version}

    def run_imaging_phase_based(self) -> dict:
        """Run imaging training (2-phase by default).

        This method is kept for backward compatibility.
        Use `run_imaging()` instead - it now also runs 2-phase training.
        """
        logger.warning(
            "run_imaging_phase_based() is deprecated. Use run_imaging() instead (now 2-phase by default)."
        )
        return self.run_imaging()

    def run(self) -> dict:
        """Run imaging training pipeline."""
        return self.run_imaging()
