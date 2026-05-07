from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImagingIngestionConfig:
    root_dir: Path
    dataset_name: str
    train_split: str
    max_samples: Optional[int]


@dataclass(frozen=True)
class ImagingCleaningConfig:
    root_dir: Path
    source_dir: Path


@dataclass(frozen=True)
class ImagingTransformationConfig:
    root_dir: Path
    source_dir: Path
    samaya_reports_csv: Path
    samaya_images_dir: Path
    image_size: int
    train_csv: Path
    val_csv: Path
    test_csv: Path
    samaya_csv: Path


@dataclass(frozen=True)
class ImagingModelTrainerConfig:
    root_dir: Path
    model_name: str
    pretrained: bool
    checkpoint_path: Path
    image_size: int


@dataclass(frozen=True)
class ImagingModelEvaluationConfig:
    root_dir: Path
    test_csv: Path
    samaya_csv: Path
    model_path: Path
    metric_file: Path
    mlflow_uri: str
    experiment_name: str
    run_name: str


@dataclass(frozen=True)
class ClinicalIngestionConfig:
    reports_csv: Path
    reports_json: Path
    images_dir: Path
    raw_csv: Path


@dataclass(frozen=True)
class ClinicalCleaningConfig:
    root_dir: Path


@dataclass(frozen=True)
class ClinicalTransformationConfig:
    root_dir: Path
    source_csv: Path
    train_csv: Path
    test_csv: Path
    feature_file: Path


@dataclass(frozen=True)
class ClinicalModelTrainerConfig:
    root_dir: Path
    model_name: str
    checkpoint_path: Path
    feature_importance_path: Path
    feature_file: Path


@dataclass(frozen=True)
class ClinicalModelEvaluationConfig:
    root_dir: Path
    test_csv: Path
    model_path: Path
    metric_file: Path
    mlflow_uri: str
    experiment_name: str
    run_name: str


@dataclass(frozen=True)
class ImagingMonitoringConfig:
    reports_dir: Path
    drift_report: Path
    classification_report: Path
    reference_csv: Path
    current_csv: Path


@dataclass(frozen=True)
class ClinicalMonitoringConfig:
    reports_dir: Path
    drift_report: Path
    classification_report: Path
    reference_csv: Path
    current_csv: Path


@dataclass(frozen=True)
class MonitoringConfig:
    reports_dir: Path
    imaging: ImagingMonitoringConfig
    clinical: ClinicalMonitoringConfig
    prometheus_port: int


@dataclass(frozen=True)
class OCRPipelineConfig:
    input_dir: Path
    output_dir: Path
    json_output: Path
    csv_output: Path
    images_dir: Path
    regions_config: dict
