"""
doc-pipeline: Pipeline de classificação e extração de dados de documentos brasileiros.
"""

# Load .env file early (before any HuggingFace imports)
from dotenv import load_dotenv
load_dotenv()

__version__ = "0.1.0"

# Registra suporte a HEIC/HEIF/AVIF (formatos de smartphone)
from pillow_heif import register_heif_opener

register_heif_opener()

# AVIF support added in pillow-heif >= 0.13.0
try:
    from pillow_heif import register_avif_opener
    register_avif_opener()
except ImportError:
    pass  # AVIF not available in this version

from .pipeline import DocumentPipeline
from .schemas import (
    CINData,
    CNHData,
    ClassificationResult,
    DocumentType,
    ExtractionResult,
    PipelineResult,
    RGData,
)

__all__ = [
    "DocumentPipeline",
    "DocumentType",
    "RGData",
    "CNHData",
    "CINData",
    "ClassificationResult",
    "ExtractionResult",
    "PipelineResult",
]
