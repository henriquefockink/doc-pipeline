"""
Schemas Pydantic para validação de dados extraídos.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Tipos de documentos suportados."""

    # RG variants
    RG_ABERTO = "rg_aberto"
    RG_DIGITAL = "rg_digital"
    RG_FRENTE = "rg_frente"
    RG_VERSO = "rg_verso"

    # CNH variants
    CNH_ABERTA = "cnh_aberta"
    CNH_DIGITAL = "cnh_digital"
    CNH_FRENTE = "cnh_frente"
    CNH_VERSO = "cnh_verso"

    @property
    def is_rg(self) -> bool:
        """Verifica se é um documento RG."""
        return self.value.startswith("rg_")

    @property
    def is_cnh(self) -> bool:
        """Verifica se é um documento CNH."""
        return self.value.startswith("cnh_")

    @property
    def base_type(self) -> str:
        """Retorna o tipo base (rg ou cnh)."""
        return "rg" if self.is_rg else "cnh"


class RGData(BaseModel):
    """Dados extraídos de um RG."""

    nome: str | None = Field(default=None, description="Nome completo")
    nome_pai: str | None = Field(default=None, description="Nome do pai")
    nome_mae: str | None = Field(default=None, description="Nome da mãe")
    data_nascimento: str | None = Field(
        default=None, description="Data de nascimento (DD/MM/AAAA)"
    )
    naturalidade: str | None = Field(
        default=None, description="Cidade/Estado de nascimento"
    )
    cpf: str | None = Field(default=None, description="CPF (###.###.###-##)")
    rg: str | None = Field(default=None, description="Número do RG")
    data_expedicao: str | None = Field(
        default=None, description="Data de expedição (DD/MM/AAAA)"
    )
    orgao_expedidor: str | None = Field(
        default=None, description="Órgão expedidor (ex: SSP-SP)"
    )


class CNHData(BaseModel):
    """Dados extraídos de uma CNH."""

    nome: str | None = Field(default=None, description="Nome completo")
    cpf: str | None = Field(default=None, description="CPF (###.###.###-##)")
    data_nascimento: str | None = Field(
        default=None, description="Data de nascimento (DD/MM/AAAA)"
    )
    numero_registro: str | None = Field(
        default=None, description="Número de registro da CNH"
    )
    numero_espelho: str | None = Field(
        default=None, description="Número do espelho da CNH"
    )
    validade: str | None = Field(default=None, description="Data de validade (DD/MM/AAAA)")
    categoria: str | None = Field(
        default=None, description="Categoria da CNH (A, B, AB, C, D, E)"
    )
    observacoes: str | None = Field(
        default=None, description="Observações/restrições"
    )
    primeira_habilitacao: str | None = Field(
        default=None, description="Data da primeira habilitação (DD/MM/AAAA)"
    )


class ImageCorrection(BaseModel):
    """Information about image preprocessing corrections applied."""

    was_corrected: bool = Field(description="Whether any correction was applied")
    rotation_applied: int = Field(
        default=0, description="Rotation applied in degrees (0, 90, 180, 270)"
    )
    correction_method: str | None = Field(
        default=None, description="Method used: 'exif', 'text_detection', or 'exif+text_detection'"
    )
    confidence: float | None = Field(
        default=None, description="Confidence of text-based detection (if used)"
    )


class ClassificationResult(BaseModel):
    """Resultado da classificação de um documento."""

    model_config = ConfigDict(frozen=True)

    document_type: DocumentType = Field(description="Tipo do documento classificado")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiança da classificação")


class ExtractionResult(BaseModel):
    """Resultado da extração de dados de um documento."""

    model_config = ConfigDict(frozen=True)

    document_type: DocumentType = Field(description="Tipo do documento")
    data: RGData | CNHData = Field(description="Dados extraídos")
    raw_text: str | None = Field(
        default=None, description="Texto bruto extraído (para debug)"
    )
    backend: str = Field(description="Backend usado para extração")


class PipelineResult(BaseModel):
    """Resultado completo do pipeline (classificação + extração)."""

    file_path: str | None = Field(default=None, description="Caminho do arquivo processado")
    classification: ClassificationResult = Field(description="Resultado da classificação")
    extraction: ExtractionResult | None = Field(
        default=None, description="Resultado da extração (None se não extraído)"
    )
    image_correction: ImageCorrection | None = Field(
        default=None, description="Info about image orientation correction (if applied)"
    )
    success: bool = Field(default=True, description="Se o processamento foi bem-sucedido")
    error: str | None = Field(default=None, description="Mensagem de erro, se houver")

    @property
    def document_type(self) -> DocumentType:
        """Atalho para o tipo do documento."""
        return self.classification.document_type

    @property
    def data(self) -> RGData | CNHData | None:
        """Atalho para os dados extraídos."""
        return self.extraction.data if self.extraction else None


# OCR Results


class OCRBoundingBox(BaseModel):
    """Bounding box for detected text."""

    x1: float
    y1: float
    x2: float
    y2: float


class OCRTextBlock(BaseModel):
    """A single detected text block with position."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: OCRBoundingBox | None = None


class OCRPageResult(BaseModel):
    """OCR result for a single page."""

    page: int = Field(ge=1, description="Page number (1-indexed)")
    text: str = Field(description="Extracted text from page")
    confidence: float = Field(ge=0.0, le=1.0, description="Average confidence")
    blocks: list[OCRTextBlock] | None = Field(
        default=None, description="Individual text blocks with positions"
    )


class OCRResult(BaseModel):
    """Complete OCR result for a document."""

    request_id: str = Field(description="Unique request identifier")
    total_pages: int = Field(ge=1, description="Total number of pages processed")
    pages: list[OCRPageResult] = Field(description="OCR results per page")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")
    file_type: str = Field(description="Detected file type (pdf, image)")
    language: str | None = Field(default=None, description="Detected or specified language")
