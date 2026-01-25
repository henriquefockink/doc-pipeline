"""
Configurações do doc-pipeline usando pydantic-settings.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractorBackend(str, Enum):
    """Backends disponíveis para extração de dados."""

    QWEN_VL = "qwen-vl"
    GOT_OCR = "got-ocr"


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
        default=ExtractorBackend.QWEN_VL,
        description="Backend para extração (qwen-vl ou got-ocr)",
    )
    extractor_device: str = Field(
        default="cuda:0",
        description="Device para o extractor (pode ser diferente do classifier)",
    )
    extractor_model_qwen: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Modelo Qwen-VL a usar",
    )
    extractor_model_got: str = Field(
        default="stepfun-ai/GOT-OCR-2.0-hf",
        description="Modelo GOT-OCR a usar",
    )

    # API settings
    api_host: str = Field(default="0.0.0.0", description="Host da API")
    api_port: int = Field(default=8001, description="Porta da API")

    # General settings
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para classificação",
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
