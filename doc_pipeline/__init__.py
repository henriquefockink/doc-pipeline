"""
doc-pipeline: Pipeline de classificação e extração de dados de documentos brasileiros.
"""

__version__ = "0.1.0"

from .pipeline import DocumentPipeline
from .schemas import (
    DocumentType,
    RGData,
    CNHData,
    ClassificationResult,
    ExtractionResult,
    PipelineResult,
)

__all__ = [
    "DocumentPipeline",
    "DocumentType",
    "RGData",
    "CNHData",
    "ClassificationResult",
    "ExtractionResult",
    "PipelineResult",
]
