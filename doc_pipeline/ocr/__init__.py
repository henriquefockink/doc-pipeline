"""OCR module using PaddleOCR."""

from .converter import PDFConverter
from .engine import OCREngine

__all__ = ["OCREngine", "PDFConverter"]
