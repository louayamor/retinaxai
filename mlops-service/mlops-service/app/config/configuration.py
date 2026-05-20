from pathlib import Path

from app.config.settings import settings
from app.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.config.config_entity import (
    ImagingCleaningConfig,
    ImagingIngestionConfig,
    ImagingModelEvaluationConfig,
    ImagingModelTrainerConfig,
    ImagingMonitoringConfig,
    ImagingTransformationConfig,
    OCRPipelineConfig,
)
from app.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
        schema_path: Path = SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)
        create_directories([settings.artifacts_root])

    def get_imaging_ingestion_config(self) -> ImagingIngestionConfig:
        cfg = self.config.data_ingestion
        root_dir = settings.artifacts_root / "data" / "raw"
        create_directories([root_dir])
        return ImagingIngestionConfig(
            root_dir=root_dir,
            dataset_name=cfg.huggingface.dataset_name,
            train_split=cfg.huggingface.train_split,
            max_samples=cfg.huggingface.get("max_samples", None),
        )

    def get_imaging_cleaning_config(self) -> ImagingCleaningConfig:
        root_dir = settings.artifacts_root / "data" / "raw"
        return ImagingCleaningConfig(root_dir=root_dir, source_dir=root_dir)

    def get_imaging_transformation_config(self) -> ImagingTransformationConfig:
        image_size = int(self.params.get("global", {}).get("image_size", 384))
        create_directories(
            [settings.imaging_data_dir, settings.imaging_train_csv.parent]
        )
        return ImagingTransformationConfig(
            root_dir=settings.imaging_data_dir,
            source_dir=settings.artifacts_root / "data" / "raw",
            samaya_reports_csv=Path(self.config.data_ingestion.samaya.reports_csv),
            samaya_images_dir=Path(self.config.data_ingestion.samaya.images_dir),
            image_size=image_size,
            train_csv=settings.imaging_train_csv,
            val_csv=settings.imaging_data_dir / "val.csv",
            test_csv=settings.imaging_test_csv,
            samaya_csv=settings.imaging_samaya_csv,
        )

    def get_imaging_model_trainer_config(self) -> ImagingModelTrainerConfig:
        create_directories([settings.imaging_artifacts_dir])
        return ImagingModelTrainerConfig(
            root_dir=settings.imaging_artifacts_dir,
            model_name=self.config.imaging_model.model_name,
            pretrained=self.config.imaging_model.pretrained,
            checkpoint_path=settings.imaging_model_path,
            image_size=int(self.params.get("global", {}).get("image_size", 384)),
        )

    def get_imaging_model_evaluation_config(self) -> ImagingModelEvaluationConfig:
        create_directories([settings.imaging_artifacts_dir])
        return ImagingModelEvaluationConfig(
            root_dir=settings.imaging_artifacts_dir,
            model_name=self.config.imaging_model.model_name,
            test_csv=settings.imaging_test_csv,
            samaya_csv=settings.imaging_samaya_csv,
            model_path=settings.imaging_model_path,
            metric_file=settings.imaging_metrics_path,
            mlflow_uri=settings.mlflow_tracking_uri,
            experiment_name=self.config.mlflow.experiment_name,
            run_name=self.config.mlflow.imaging_run_name,
        )

    def get_monitoring_config(self) -> ImagingMonitoringConfig:
        create_directories([settings.monitoring_dir])
        return ImagingMonitoringConfig(
            reports_dir=settings.monitoring_dir,
            drift_report=settings.monitoring_dir / "imaging_drift_report.html",
            classification_report=settings.monitoring_dir
            / "imaging_classification_report.html",
            reference_csv=settings.imaging_train_csv,
            current_csv=settings.imaging_test_csv,
        )

    def get_ocr_pipeline_config(self) -> OCRPipelineConfig:
        cfg = self.config.ocr_pipeline
        regions = read_yaml(Path(cfg.regions_config))
        create_directories([Path(cfg.output_dir), Path(cfg.images_dir)])
        return OCRPipelineConfig(
            input_dir=Path(cfg.input_dir),
            output_dir=Path(cfg.output_dir),
            json_output=Path(cfg.json_output),
            csv_output=Path(cfg.csv_output),
            images_dir=Path(cfg.images_dir),
            regions_config=regions.get("regions", {}),
        )
