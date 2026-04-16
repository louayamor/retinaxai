from enum import StrEnum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


def _get_service_root() -> Path:
    """Get this service's root directory (where this config file lives).

    If RETINAXAI_BASE_DIR is set, use that as base; otherwise derive from __file__.
    """
    import os

    base_dir = os.environ.get("RETINAXAI_BASE_DIR")
    if base_dir:
        return Path(base_dir) / "llmops-service" / "llmops-service"
    return Path(__file__).parent.parent


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GITHUB = "github"
    MOCK = "mock"


class Settings(BaseSettings):
    app_name: str = "RetinaXAI LLMOps Service"
    app_version: str = "0.1.0"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8002

    cors_origins: list[str] = ["http://localhost:3000"]

    backend_service_url: str = Field(
        default="http://backend-service:8000", validation_alias="BACKEND_SERVICE_URL"
    )
    mlops_service_url: str = Field(
        default="http://mlops-service:8004", validation_alias="MLOPS_SERVICE_URL"
    )
    timeout_seconds: int = 120
    max_tokens: int = Field(default=2000, validation_alias="LLM_MAX_TOKENS")
    rag_manifest_url: str = Field(
        default="http://mlops-service:8004/rag/manifest",
        validation_alias="RAG_MANIFEST_URL",
    )

    llm_provider: LLMProvider = LLMProvider.GITHUB
    llm_model: str = "openai/gpt-4.1-mini"
    llm_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    llm_base_url: Optional[str] = None

    github_token: Optional[str] = Field(
        default=None, validation_alias="GITHUB_ACCESS_TOKEN"
    )
    github_endpoint: str = "https://models.github.ai/inference"

    api_key: str = Field(default="", validation_alias="LLMOPS_API_KEY")
    backend_api_key: str = Field(default="", validation_alias="BACKEND_API_KEY")

    rate_limit_max_requests: int = 100
    rate_limit_window_seconds: int = 60
    enable_rate_limiting: bool = True

    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: str = ""
    mlflow_experiment_name: str = "retinaxai-llmops"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"

    prometheus_metrics_port: int = 9092

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def data_dir(self) -> Path:
        """Service-relative data directory.

        If RETINAXAI_BASE_DIR is set, use that as base; otherwise derive from __file__.
        """
        import os

        base_dir = os.environ.get("RETINAXAI_BASE_DIR")
        if base_dir:
            return Path(base_dir) / "data"
        return _get_service_root() / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def rag_chroma_persist_directory(self) -> Path:
        return self.data_dir / "rag" / "chroma"

    @property
    def rag_chroma_collection_name(self) -> str:
        return "retinaxai_rag"

    @property
    def rag_embedding_model(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def rag_chunk_size(self) -> int:
        return 800

    @property
    def rag_chunk_overlap(self) -> int:
        return 80

    @property
    def artifacts_root(self) -> Path:
        """Path to MLOps artifacts (models, data)."""
        import os

        base_dir = os.environ.get("RETINAXAI_BASE_DIR")
        if base_dir:
            return Path(base_dir) / "mlops-service" / "mlops-service" / "artifacts"
        return _get_service_root() / "artifacts"

    @property
    def clinical_model_path(self) -> Path:
        return self.artifacts_root / "model" / "clinical" / "model.pkl"

    @property
    def imaging_model_path(self) -> Path:
        return self.artifacts_root / "model" / "imaging" / "model.pth"


settings = Settings()
