"""
Extractors module - backends VLM para extração de dados.
"""

from .base import BaseExtractor
from .hybrid import HybridExtractor
from .qwen_vl import QwenVLExtractor
from .vllm_client import VLLMClient
from .vllm_embedded import VLLMEmbeddedClient


# EasyOCRExtractor requires easyocr, import lazily to avoid dependency in API
def __getattr__(name):
    if name == "EasyOCRExtractor":
        from .easyocr import EasyOCRExtractor

        return EasyOCRExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseExtractor",
    "EasyOCRExtractor",
    "HybridExtractor",
    "QwenVLExtractor",
    "VLLMClient",
    "VLLMEmbeddedClient",
]
