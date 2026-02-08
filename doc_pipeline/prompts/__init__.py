"""
Prompts para extração de dados de documentos.
"""

from .cin import CIN_EXTRACTION_PROMPT, CIN_FIELDS
from .cnh import CNH_EXTRACTION_PROMPT, CNH_FIELDS
from .rg import RG_EXTRACTION_PROMPT, RG_FIELDS

__all__ = [
    "CNH_EXTRACTION_PROMPT",
    "CNH_FIELDS",
    "CIN_EXTRACTION_PROMPT",
    "CIN_FIELDS",
    "RG_EXTRACTION_PROMPT",
    "RG_FIELDS",
]
