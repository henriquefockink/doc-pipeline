"""
Extractors module - backends VLM para extração de dados.
"""

from .base import BaseExtractor
from .qwen_vl import QwenVLExtractor
from .easyocr import EasyOCRExtractor

__all__ = ["BaseExtractor", "QwenVLExtractor", "EasyOCRExtractor"]
