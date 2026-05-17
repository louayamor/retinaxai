from loguru import logger
from pathlib import Path
from app.constants import PARAMS_FILE_PATH
from app.utils.common import read_yaml
from app.domains.imaging.pipeline.stage_01_data_ingestion import run as imaging_ingest
from app.domains.imaging.pipeline.stage_02_data_cleaning import run as imaging_clean
from app.domains.imaging.pipeline.stage_03_data_transformation import (
    run as imaging_transform,
)
from app.domains.imaging.pipeline.stage_04_model_trainer import run as imaging_train
from app.domains.imaging.pipeline.stage_05_model_evaluation import (
    run as imaging_evaluate,
)
from app.domains.clinical.pipeline.stage_01_data_ingestion import run as clinical_ingest
from app.domains.clinical.pipeline.stage_02_data_cleaning import run as clinical_clean
from app.domains.clinical.pipeline.stage_03_data_transformation import (
    run as clinical_transform,
)
from app.domains.clinical.pipeline.stage_04_model_trainer import run as clinical_train
from app.domains.clinical.pipeline.stage_05_model_evaluation import (
    run as clinical_evaluate,
)
from app.services.registry.model_registry import ModelRegistryService
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
            return "v1.0.0"

        latest = sorted(existing, key=lambda v: v.created_at)[-1]
        major, minor, patch = latest.version.lstrip("v").split(".")
        new_minor = str(int(minor) + 1)
        return f"v{major}.{new_minor}.0"

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
            elif pipeline == "clinical":
                model_metrics = {
                    "accuracy": metrics.get("accuracy", 0),
                    "quadratic_weighted_kappa": metrics.get(
                        "quadratic_weighted_kappa", 0
                    ),
                    "roc_auc_macro": metrics.get("roc_auc_macro", 0),
                    "macro_f1": metrics.get("macro_f1", 0),
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

        except Exception as e:
            logger.error(f"Failed to register {pipeline} model: {e}")
            # Non-critical error - don't fail training if registration fails

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

    def run_clinical(self) -> dict:
        """Run clinical training pipeline."""
        logger.info("=== clinical pipeline started ===")

        clinical_ingest()
        clinical_clean()
        clinical_transform()
        clinical_train()

        # Get metrics from evaluation
        metrics = clinical_evaluate()

        # Register the trained model
        model_path = settings.clinical_model_path
        version = self._generate_version("clinical")
        self._register_model(
            pipeline="clinical", version=version, model_path=model_path, metrics=metrics
        )

        logger.info("=== clinical pipeline complete ===")
        return {"pipeline": "clinical", "metrics": metrics, "version": version}

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
        """Run unified training pipeline."""
        logger.info("=== unified training pipeline started ===")

        imaging_result = self.run_imaging()
        clinical_result = self.run_clinical()

        logger.info("=== unified training pipeline complete ===")

        return {"imaging": imaging_result, "clinical": clinical_result}
