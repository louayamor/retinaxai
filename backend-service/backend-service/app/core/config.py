from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _get_service_root() -> Path:
    """Get this service's root directory (where this config file lives)."""
    return Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "production"
    APP_NAME: str = "retinaxai-backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    SQLALCHEMY_ECHO: bool = False

    DATABASE_URL: str = "postgresql+asyncpg://louay:louay@localhost:5432/retinaxai_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40

    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    ML_SERVICE_URL: str = "http://localhost:8004"
    ML_SERVICE_TIMEOUT: int = 30
    ML_SERVICE_API_KEY: str = ""

    LLM_SERVICE_URL: str = "http://localhost:8002"
    LLM_SERVICE_TIMEOUT: int = 60
    LLM_SERVICE_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"

    REDIS_URL: str = "redis://localhost:6379"
    WS_URL: str = "ws://localhost:8000/ws"

    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    RATE_LIMIT_MAX_REQUESTS: int = 10
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            import json

            return json.loads(v)
        return v

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
    def imaging_metrics(self) -> Path:
        return self.data_dir / "metrics" / "imaging" / "metrics.json"

    @property
    def clinical_metrics(self) -> Path:
        return self.data_dir / "metrics" / "clinical" / "metrics.json"

    def ensure_dirs(self) -> None:
        for dir_path in [self.fundus_dir, self.oct_dir, self.gradcam_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
