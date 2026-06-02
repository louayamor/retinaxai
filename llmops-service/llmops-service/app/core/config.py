from enum import StrEnum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _get_service_root() -> Path:
    """Get this service's root directory (where this config file lives)."""
    return Path(__file__).parent.parent.parent


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GITHUB = "github"
    NVIDIA = "nvidia"
    MOCK = "mock"


class Settings(BaseSettings):
    app_name: str = "RetinaXAI LLMOps Service"
    app_version: str = "0.1.0"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8080, validation_alias="PORT")

    cors_origins: list[str] = ["http://localhost:3000"]

    backend_service_url: str = Field(
        default="http://backend-service:8000", validation_alias="BACKEND_SERVICE_URL"
    )
    mlops_service_url: str = Field(
        default="http://mlops-service:8004", validation_alias="MLOPS_SERVICE_URL"
    )
    mlops_model_download_url: str = Field(
        default="http://localhost:8004", validation_alias="MLOPS_MODEL_DOWNLOAD_URL"
    )
    timeout_seconds: int = 120
    max_tokens: int = Field(default=2000, validation_alias="LLM_MAX_TOKENS")
    rag_manifest_url: str = Field(
        default="http://mlops-service:8004/rag/manifest",
        validation_alias="RAG_MANIFEST_URL",
    )

    llm_provider: LLMProvider = LLMProvider.GITHUB
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    llm_base_url: Optional[str] = None

    _PROVIDER_MODELS: dict[str, str] = {
        "github": "gpt-4o",
        "nvidia": "meta/llama-3.1-8b-instruct",
    }

    @property
    def resolved_model(self) -> str:
        provider = (
            self.llm_provider.value
            if isinstance(self.llm_provider, LLMProvider)
            else str(self.llm_provider)
        )
        return self._PROVIDER_MODELS.get(provider, self.llm_model)

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
    ollama_fallback_model: str = "qwen2.5:3b"

    nvidia_api_key: Optional[str] = Field(
        default=None, validation_alias="NVIDIA_API_KEY"
    )
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    prometheus_metrics_port: int = 9092

    # ChromaDB remote HTTP client (production) — when set, overrides local persist_directory.
    chroma_host: Optional[str] = Field(
        default=None, validation_alias="CHROMA_HOST"
    )
    chroma_port: int = Field(default=8000, validation_alias="CHROMA_PORT")

    # RAG embeddings: support offline / local-only environments.
    rag_embeddings_offline: bool = Field(
        default=False,
        validation_alias="RAG_EMBEDDINGS_OFFLINE",
        description="If true, never hit HuggingFace Hub; require local cache/model path.",
    )
    rag_embedding_model_path: Optional[Path] = Field(
        default=None,
        validation_alias="RAG_EMBEDDING_MODEL_PATH",
        description="Optional local path to the embedding model directory.",
    )
    rag_hf_home: Optional[Path] = Field(
        default=None,
        validation_alias="RAG_HF_HOME",
        description="Optional HF_HOME override for embedding model cache.",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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
    def resolved_rag_embedding_model(self) -> str:
        """Resolve embedding model to local path when configured."""
        if self.rag_embedding_model_path and self.rag_embedding_model_path.exists():
            return str(self.rag_embedding_model_path)
        return self.rag_embedding_model

    @property
    def rag_chunk_size(self) -> int:
        return 800

    @property
    def rag_chunk_overlap(self) -> int:
        return 80

    @property
    def artifacts_root(self) -> Path:
        """Path to local cache for MLOps artifacts.

        When models are fetched from MLOps, they're cached locally here.
        """
        return _get_service_root() / "data" / "models"

    @property
    def clinical_model_path(self) -> Path:
        return self.artifacts_root / "clinical" / "model.pkl"

    @property
    def imaging_model_path(self) -> Path:
        return self.artifacts_root / "imaging" / "model.pth"

    async def ensure_clinical_model(self) -> Path:
        """Fetch clinical model from MLOps if not cached locally."""
        import os
        import httpx

        path = self.clinical_model_path
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.mlops_model_download_url}/models/download/clinical/model"
            )
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                return path

        raise RuntimeError(
            f"Could not fetch clinical model from {self.mlops_model_download_url}"
        )


settings = Settings()
