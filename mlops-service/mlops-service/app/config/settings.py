from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _get_service_root() -> Path:
    """Get this service's root directory (where this config file lives)."""
    return Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RetinaXAI MLOps Service"
    app_version: str = "0.1.0"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8004

    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""

    dagshub_repo_owner: str = "louayamor"
    dagshub_repo_name: str = "retinaxai"

    llmops_service_url: str = "http://llmops-service:8002"
    backend_service_url: str = "http://backend-service:8000"
    timeout_seconds: int = 30

    prometheus_metrics_port: int = 9101
    prometheus_url: str = "http://localhost:9090"
    automation_enabled: bool = False
    automation_interval_hours: int = 24
    max_training_jobs: int = 2
    max_training_jobs_per_pipeline: int = 1
    retrain_imaging_metric: str = "quadratic_weighted_kappa"
    retrain_min_improvement: float = 0.01
    retrain_cooldown_hours: int = 24

    frontend_url: str = "http://localhost:3000"
    redis_url: str = "redis://localhost:6379"
    mlops_monitor_channel: str = "mlops.monitor"

    @property
    def data_dir(self) -> Path:
        """Service-relative data directory."""
        return _get_service_root() / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def fundus_dir(self) -> Path:
        return self.upload_dir / "fundus"

    @property
    def oct_dir(self) -> Path:
        return self.upload_dir / "oct"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def gradcam_dir(self) -> Path:
        return self.output_dir / "gradcam"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def artifacts_root(self) -> Path:
        """Artifacts directory - sibling to mlops-service/."""
        return _get_service_root().parent / "artifacts"

    @property
    def imaging_artifacts_dir(self) -> Path:
        return self.artifacts_root / "model" / "imaging"

    @property
    def imaging_model_path(self) -> Path:
        return self.imaging_artifacts_dir / "model.pth"

    @property
    def evidently_metrics_path(self) -> Path:
        return self.monitoring_dir / "evidently_metrics.json"

    @property
    def imaging_metrics_path(self) -> Path:
        return self.imaging_artifacts_dir / "metrics.json"

    @property
    def imaging_data_dir(self) -> Path:
        return self.artifacts_root / "data" / "processed" / "imaging"

    @property
    def monitoring_dir(self) -> Path:
        return self.artifacts_root / "monitoring"

    @property
    def training_jobs_file(self) -> Path:
        return self.artifacts_root / "training_jobs.json"

    @property
    def model_registry_dir(self) -> Path:
        """Model registry directory for versioning and metadata."""
        return self.artifacts_root / "model_registry"

    @property
    def ocr_input_dir(self) -> Path:
        return self.data_dir / "ocr_reports"

    @property
    def ocr_output_dir(self) -> Path:
        return self.artifacts_root / "ocr" / "output"

    @property
    def imaging_train_csv(self) -> Path:
        return self.imaging_data_dir / "train.csv"

    @property
    def imaging_test_csv(self) -> Path:
        return self.imaging_data_dir / "test.csv"

    @property
    def imaging_samaya_csv(self) -> Path:
        return self.imaging_data_dir / "samaya.csv"


settings = Settings()
