"""
Configurações do doc-pipeline usando pydantic-settings.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractorBackend(StrEnum):
    """Backends disponíveis para extração de dados."""

    QWEN_VL = "qwen-vl"
    EASY_OCR = "easy-ocr"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """Configurações do pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="DOC_PIPELINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Classifier settings
    classifier_model_path: Path = Field(
        default=Path("models/classifier.pth"),
        description="Caminho para o modelo do classificador",
    )
    classifier_model_type: str = Field(
        default="efficientnet_b0",
        description="Tipo do modelo EfficientNet (b0, b2, b4)",
    )
    classifier_device: str = Field(
        default="cuda:0",
        description="Device para o classificador (cuda:0, cuda:1, cpu)",
    )
    classifier_fp8: bool = Field(
        default=False,
        description="Usar quantização FP8 (requer GPU Hopper/Blackwell)",
    )

    # Extractor settings
    extractor_backend: ExtractorBackend = Field(
        default=ExtractorBackend.HYBRID,
        description="Backend para extração (qwen-vl, easy-ocr ou hybrid)",
    )
    extractor_device: str = Field(
        default="cuda:0",
        description="Device para o extractor (pode ser diferente do classifier)",
    )
    extractor_model_qwen: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Modelo Qwen-VL a usar",
    )

    # Orientation correction settings
    orientation_enabled: bool = Field(
        default=True,
        description="Enable orientation correction via EasyOCR text-box detection",
    )
    orientation_device: str | None = Field(
        default=None,
        description="Device for orientation model (default: uses classifier_device)",
    )
    orientation_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to apply orientation correction",
    )

    # API settings
    api_host: str = Field(default="0.0.0.0", description="Host da API")
    api_port: int = Field(default=9000, description="Porta da API")

    # Authentication settings
    api_key: str | None = Field(default=None, description="API key master (env)")
    api_keys: str | None = Field(
        default=None, description="Lista de API keys master separadas por vírgula"
    )
    database_url: str | None = Field(
        default=None, description="PostgreSQL URL para API keys dinâmicas"
    )

    @property
    def api_keys_list(self) -> list[str]:
        """Retorna lista de API keys do ambiente."""
        keys = []
        if self.api_key:
            keys.append(self.api_key)
        if self.api_keys:
            keys.extend([k.strip() for k in self.api_keys.split(",") if k.strip()])
        return keys

    # General settings
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para classificação",
    )
    warmup_on_start: bool = Field(
        default=True,
        description="Carregar modelos na inicialização (warmup)",
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Nível de log (DEBUG, INFO, WARNING, ERROR)",
    )
    log_json: bool = Field(
        default=True,
        description="Logs em formato JSON (True) ou colorido (False)",
    )

    # Sentry / GlitchTip settings
    sentry_dsn: str | None = Field(default=None, description="Sentry/GlitchTip DSN")
    sentry_environment: str = Field(default="production", description="Environment tag")
    sentry_traces_sample_rate: float = Field(default=0.1, description="Transaction sampling rate")

    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_max_connections: int = Field(
        default=200,
        description="Maximum Redis connections in pool",
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    rate_limit_requests: int = Field(
        default=30,
        description="Max requests per time window",
    )
    rate_limit_window: str = Field(
        default="second",
        description="Time window for rate limit (second, minute, hour)",
    )

    # Queue settings
    max_queue_size: int = Field(
        default=500,
        description="Maximum jobs allowed in queue",
    )
    queue_timeout_seconds: float = Field(
        default=300.0,
        description="Queue operation timeout (5 min)",
    )
    sync_timeout_seconds: float = Field(
        default=300.0,
        description="Sync mode wait timeout (5 min)",
    )

    # Webhook settings
    webhook_timeout_seconds: float = Field(
        default=20.0,
        description="Webhook delivery timeout",
    )
    webhook_max_retries: int = Field(
        default=3,
        description="Maximum webhook delivery retries",
    )

    # Worker settings
    worker_health_port: int = Field(
        default=9010,
        description="Worker health check port",
    )
    worker_ocr_health_port: int = Field(
        default=9011,
        description="OCR Worker health check port",
    )

    # Inference server settings
    inference_server_enabled: bool = Field(
        default=False,
        description="Use remote inference server instead of local VLM",
    )
    inference_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for inference requests",
    )
    inference_server_health_port: int = Field(
        default=9020,
        description="Inference server health check port",
    )
    inference_batch_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max batch size for inference server (higher = more GPU throughput)",
    )
    inference_batch_timeout_ms: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Max ms to wait for batch to fill before processing",
    )

    # Worker concurrency settings
    worker_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Max concurrent jobs per worker (>1 enables batching with inference server)",
    )

    # OCR settings
    ocr_language: str = Field(
        default="pt",
        description="OCR language (pt, en, ch, etc.)",
    )
    ocr_use_gpu: bool = Field(
        default=True,
        description="Use GPU for OCR (PaddleOCR)",
    )
    ocr_max_pages: int = Field(
        default=10,
        description="Maximum pages to process from PDF",
    )

    @field_validator("classifier_model_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Resolve o caminho para Path absoluto."""
        return Path(v).expanduser().resolve()


# Singleton para configurações globais
_settings: Settings | None = None


def get_settings() -> Settings:
    """Retorna as configurações do pipeline (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reseta as configurações (útil para testes)."""
    global _settings
    _settings = None
