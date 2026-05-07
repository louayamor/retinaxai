import os
from pathlib import Path

from app.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.entity.config_entity import (
    ClinicalCleaningConfig,
    ClinicalIngestionConfig,
    ClinicalModelEvaluationConfig,
    ClinicalModelTrainerConfig,
    ClinicalMonitoringConfig,
    ClinicalTransformationConfig,
    ImagingCleaningConfig,
    ImagingIngestionConfig,
    ImagingModelEvaluationConfig,
    ImagingModelTrainerConfig,
    ImagingMonitoringConfig,
    ImagingTransformationConfig,
    MonitoringConfig,
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
        create_directories([Path(self.config.artifacts_root)])

    def get_imaging_ingestion_config(self) -> ImagingIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([Path(cfg.root_dir)])
        return ImagingIngestionConfig(
            root_dir=Path(cfg.root_dir),
            dataset_name=cfg.huggingface.dataset_name,
            train_split=cfg.huggingface.train_split,
            max_samples=cfg.huggingface.get("max_samples", None),
        )

    def get_imaging_cleaning_config(self) -> ImagingCleaningConfig:
        cfg = self.config.data_ingestion
        return ImagingCleaningConfig(
            root_dir=Path(cfg.root_dir),
            source_dir=Path(cfg.root_dir),
        )

    def get_imaging_transformation_config(self) -> ImagingTransformationConfig:
        cfg = self.config.data_transformation.imaging
        samaya_cfg = self.config.data_ingestion.samaya
        create_directories(
            [
                Path(cfg.root_dir),
                Path(cfg.train_csv).parent,
            ]
        )
        return ImagingTransformationConfig(
            root_dir=Path(cfg.root_dir),
            source_dir=Path(self.config.data_ingestion.root_dir),
            samaya_reports_csv=Path(samaya_cfg.reports_csv),
            samaya_images_dir=Path(samaya_cfg.images_dir),
            image_size=cfg.image_size,
            train_csv=Path(cfg.train_csv),
            val_csv=Path(cfg.val_csv),
            test_csv=Path(cfg.test_csv),
            samaya_csv=Path(cfg.samaya_csv),
        )

    def get_imaging_model_trainer_config(self) -> ImagingModelTrainerConfig:
        cfg = self.config.imaging_model
        params = read_yaml("config/params.yaml")
        create_directories([Path(cfg.root_dir)])
        return ImagingModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            checkpoint_path=Path(cfg.checkpoint_path),
            image_size=params.dl_training.image_size,
        )

    def get_imaging_model_evaluation_config(self) -> ImagingModelEvaluationConfig:
        cfg = self.config.model_evaluation.imaging
        mlflow_cfg = self.config.mlflow
        create_directories([Path(cfg.root_dir)])
        return ImagingModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            test_csv=Path(cfg.test_csv),
            samaya_csv=Path(self.config.data_transformation.imaging.samaya_csv),
            model_path=Path(cfg.model_path),
            metric_file=Path(cfg.metric_file),
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
            experiment_name=mlflow_cfg.experiment_name,
            run_name=mlflow_cfg.imaging_run_name,
        )

    def get_clinical_ingestion_config(self) -> ClinicalIngestionConfig:
        cfg = self.config.data_ingestion.samaya
        trans_cfg = self.config.data_transformation.clinical
        return ClinicalIngestionConfig(
            reports_csv=Path(cfg.reports_csv),
            reports_json=Path(cfg.reports_json),
            images_dir=Path(cfg.images_dir),
            raw_csv=Path(trans_cfg.raw_csv),
        )

    def get_clinical_cleaning_config(self) -> ClinicalCleaningConfig:
        cfg = self.config.data_transformation.clinical
        return ClinicalCleaningConfig(
            root_dir=Path(cfg.root_dir),
        )

    def get_clinical_transformation_config(self) -> ClinicalTransformationConfig:
        cfg = self.config.data_transformation.clinical
        create_directories(
            [
                Path(cfg.root_dir),
                Path(cfg.train_csv).parent,
            ]
        )
        return ClinicalTransformationConfig(
            root_dir=Path(cfg.root_dir),
            source_csv=Path(self.config.data_ingestion.samaya.reports_csv),
            train_csv=Path(cfg.train_csv),
            test_csv=Path(cfg.test_csv),
            feature_file=Path(cfg.feature_file),
        )

    def get_clinical_model_trainer_config(self) -> ClinicalModelTrainerConfig:
        cfg = self.config.clinical_model
        trans_cfg = self.config.data_transformation.clinical
        create_directories([Path(cfg.root_dir)])
        return ClinicalModelTrainerConfig(
            root_dir=Path(cfg.root_dir),
            model_name=cfg.model_name,
            checkpoint_path=Path(cfg.checkpoint_path),
            feature_importance_path=Path(cfg.feature_importance_path),
            feature_file=Path(trans_cfg.feature_file),
        )

    def get_clinical_model_evaluation_config(self) -> ClinicalModelEvaluationConfig:
        cfg = self.config.model_evaluation.clinical
        mlflow_cfg = self.config.mlflow
        create_directories([Path(cfg.root_dir)])
        return ClinicalModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            test_csv=Path(cfg.test_csv),
            model_path=Path(cfg.model_path),
            metric_file=Path(cfg.metric_file),
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
            experiment_name=mlflow_cfg.experiment_name,
            run_name=mlflow_cfg.clinical_run_name,
        )

    def get_monitoring_config(self) -> MonitoringConfig:
        cfg = self.config.monitoring
        create_directories([Path(cfg.reports_dir)])
        return MonitoringConfig(
            reports_dir=Path(cfg.reports_dir),
            imaging=ImagingMonitoringConfig(
                reports_dir=Path(cfg.reports_dir),
                drift_report=Path(cfg.imaging.drift_report),
                classification_report=Path(cfg.imaging.classification_report),
                reference_csv=Path(cfg.imaging.reference_csv),
                current_csv=Path(cfg.imaging.current_csv),
            ),
            clinical=ClinicalMonitoringConfig(
                reports_dir=Path(cfg.reports_dir),
                drift_report=Path(cfg.clinical.drift_report),
                classification_report=Path(cfg.clinical.classification_report),
                reference_csv=Path(cfg.clinical.reference_csv),
                current_csv=Path(cfg.clinical.current_csv),
            ),
            prometheus_port=cfg.prometheus_port,
        )

    def get_ocr_pipeline_config(self) -> OCRPipelineConfig:
        cfg = self.config.ocr_pipeline
        regions = read_yaml(Path(cfg.regions_config))
        create_directories(
            [
                Path(str(cfg.output_dir)),
                Path(str(cfg.images_dir)),
            ]
        )
        return OCRPipelineConfig(
            input_dir=Path(str(cfg.input_dir)),
            output_dir=Path(str(cfg.output_dir)),
            json_output=Path(str(cfg.json_output)),
            csv_output=Path(str(cfg.csv_output)),
            images_dir=Path(str(cfg.images_dir)),
            regions_config=regions.get("regions", {}),
        )
