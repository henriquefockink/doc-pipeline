"""
Interface abstrata para extractors de dados de documentos.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from ..schemas import (
    CNHData,
    DocumentType,
    ExtractionResult,
    GenericExtractionResult,
    GenericPageData,
    RGData,
)


class BaseExtractor(ABC):
    """Interface base para extractors de dados de documentos."""

    backend_name: str = "base"
    supports_pdf: bool = False  # Override em subclasses que suportam PDF nativo

    @abstractmethod
    def load_model(self) -> None:
        """Carrega o modelo VLM."""

    @abstractmethod
    def unload_model(self) -> None:
        """Descarrega o modelo para liberar memória."""

    @abstractmethod
    def extract_text(self, image: str | Path | Image.Image) -> str:
        """
        Extrai texto bruto da imagem.

        Args:
            image: Caminho da imagem ou PIL.Image

        Returns:
            Texto extraído da imagem
        """

    def extract_text_from_pdf(self, pdf_path: str | Path) -> list[str]:
        """
        Extrai texto de cada página de um PDF.

        Args:
            pdf_path: Caminho do arquivo PDF

        Returns:
            Lista de textos, um por página

        Raises:
            NotImplementedError: Se o extractor não suporta PDF nativo
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} não suporta PDF nativo. "
            "Use um extractor com supports_pdf=True ou converta para imagem."
        )

    @abstractmethod
    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """
        Extrai dados de um RG.

        Args:
            image: Imagem do RG

        Returns:
            RGData com campos extraídos
        """

    @abstractmethod
    def extract_cnh(self, image: str | Path | Image.Image) -> CNHData:
        """
        Extrai dados de uma CNH.

        Args:
            image: Imagem da CNH

        Returns:
            CNHData com campos extraídos
        """

    def extract(
        self, image: str | Path | Image.Image, document_type: DocumentType
    ) -> ExtractionResult:
        """
        Extrai dados de um documento baseado no tipo.

        Args:
            image: Imagem do documento
            document_type: Tipo do documento (RG, CNH ou GENERIC)

        Returns:
            ExtractionResult com dados extraídos
        """
        if document_type.is_generic:
            raise ValueError(
                "Use extract_generic() para documentos genéricos, "
                "ou extract_generic_from_pdf() para PDFs."
            )

        if document_type.is_rg:
            data = self.extract_rg(image)
        elif document_type.is_cnh:
            data = self.extract_cnh(image)
        else:
            raise ValueError(f"Tipo de documento não suportado: {document_type}")

        return ExtractionResult(
            document_type=document_type,
            data=data,
            backend=self.backend_name,
        )

    def extract_generic(
        self, image: str | Path | Image.Image
    ) -> GenericExtractionResult:
        """
        Extrai texto bruto de uma imagem (OCR puro).

        Args:
            image: Imagem do documento

        Returns:
            GenericExtractionResult com texto extraído
        """
        text = self.extract_text(image)
        return GenericExtractionResult(
            document_type=DocumentType.GENERIC,
            raw_text=text,
            pages=[GenericPageData(page=1, text=text)],
            total_pages=1,
            backend=self.backend_name,
        )

    def extract_generic_from_pdf(
        self, pdf_path: str | Path
    ) -> GenericExtractionResult:
        """
        Extrai texto bruto de um PDF (OCR puro, multi-page).

        Args:
            pdf_path: Caminho do arquivo PDF

        Returns:
            GenericExtractionResult com texto de todas as páginas
        """
        page_texts = self.extract_text_from_pdf(pdf_path)

        pages = [
            GenericPageData(page=i + 1, text=text)
            for i, text in enumerate(page_texts)
        ]

        return GenericExtractionResult(
            document_type=DocumentType.GENERIC,
            raw_text="\n\n".join(page_texts),
            pages=pages,
            total_pages=len(pages),
            backend=self.backend_name,
        )

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        """Helper para carregar imagem."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")
