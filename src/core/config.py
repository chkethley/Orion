"""Configuration management for Orion AGI."""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Application
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/orion_agi",
        alias="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Model Settings
    default_model: str = Field(default="gpt-4", alias="DEFAULT_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL"
    )
    max_tokens: int = Field(default=4096, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # Experiment Tracking
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI"
    )
    wandb_api_key: Optional[str] = Field(default=None, alias="WANDB_API_KEY")
    wandb_project: str = Field(default="orion-agi", alias="WANDB_PROJECT")

    # Vector Database
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(
        default=None,
        alias="PINECONE_ENVIRONMENT"
    )
    chromadb_host: str = Field(default="localhost", alias="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, alias="CHROMADB_PORT")

    # Compute Resources
    cuda_visible_devices: str = Field(default="0", alias="CUDA_VISIBLE_DEVICES")
    torch_device: str = Field(default="cuda", alias="TORCH_DEVICE")
    num_workers: int = Field(default=4, alias="NUM_WORKERS")

    # API Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")

    # Security
    secret_key: str = Field(
        default="your_secret_key_here",
        alias="SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        alias="CELERY_RESULT_BACKEND"
    )


# Global settings instance
settings = Settings()
