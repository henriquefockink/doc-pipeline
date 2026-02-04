"""OCR module using EasyOCR."""

from .engine import OCREngine

# PDFConverter requires PyMuPDF, import lazily to avoid dependency in API
def __getattr__(name):
    if name == "PDFConverter":
        from .converter import PDFConverter
        return PDFConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["OCREngine", "PDFConverter"]
