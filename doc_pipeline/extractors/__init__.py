"""
Extractors module - backends VLM para extração de dados.
"""

from .base import BaseExtractor
from .qwen_vl import QwenVLExtractor
from .got_ocr import GOTOCRExtractor

__all__ = ["BaseExtractor", "QwenVLExtractor", "GOTOCRExtractor"]
