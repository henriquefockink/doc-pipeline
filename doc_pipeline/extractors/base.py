"""
Interface abstrata para extractors de dados de documentos.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from ..schemas import CNHData, DocumentType, ExtractionResult, RGData


class BaseExtractor(ABC):
    """Interface base para extractors de dados de documentos."""

    backend_name: str = "base"

    @abstractmethod
    def load_model(self) -> None:
        """Carrega o modelo VLM."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Descarrega o modelo para liberar memória."""
        pass

    @abstractmethod
    def extract_text(self, image: str | Path | Image.Image) -> str:
        """
        Extrai texto bruto da imagem.

        Args:
            image: Caminho da imagem ou PIL.Image

        Returns:
            Texto extraído da imagem
        """
        pass

    @abstractmethod
    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """
        Extrai dados de um RG.

        Args:
            image: Imagem do RG

        Returns:
            RGData com campos extraídos
        """
        pass

    @abstractmethod
    def extract_cnh(self, image: str | Path | Image.Image) -> CNHData:
        """
        Extrai dados de uma CNH.

        Args:
            image: Imagem da CNH

        Returns:
            CNHData com campos extraídos
        """
        pass

    def extract(
        self, image: str | Path | Image.Image, document_type: DocumentType
    ) -> ExtractionResult:
        """
        Extrai dados de um documento baseado no tipo.

        Args:
            image: Imagem do documento
            document_type: Tipo do documento (RG ou CNH)

        Returns:
            ExtractionResult com dados extraídos
        """
        if document_type.is_rg:
            data = self.extract_rg(image)
        elif document_type.is_cnh:
            data = self.extract_cnh(image)
        else:
            raise ValueError(f"Tipo de documento não suportado: {document_type}")

        return ExtractionResult(
            document_type=document_type,
            data=data,
            backend="paneas_v1",
        )

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        """Helper para carregar imagem."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")
