"""
Orquestrador principal do pipeline de documentos.
"""

from pathlib import Path
from typing import Iterator

from PIL import Image

from .classifier import ClassifierAdapter
from .config import ExtractorBackend, Settings, get_settings
from .extractors import BaseExtractor, GOTOCRExtractor, QwenVLExtractor
from .schemas import (
    ClassificationResult,
    DocumentType,
    ExtractionResult,
    GenericExtractionResult,
    PipelineResult,
)


class DocumentPipeline:
    """Pipeline completo para classificação e extração de documentos."""

    def __init__(
        self,
        classifier_model_path: str | Path | None = None,
        classifier_model_type: str | None = None,
        classifier_device: str | None = None,
        classifier_fp8: bool | None = None,
        extractor_backend: ExtractorBackend | str | None = None,
        extractor_device: str | None = None,
        settings: Settings | None = None,
    ):
        """
        Inicializa o pipeline.

        Args:
            classifier_model_path: Caminho para o modelo do classificador
            classifier_model_type: Tipo do modelo (efficientnet_b0, b2, b4)
            classifier_device: Device para o classificador
            classifier_fp8: Usar FP8 no classificador
            extractor_backend: Backend para extração (qwen-vl ou got-ocr)
            extractor_device: Device para o extractor
            settings: Settings customizadas (sobrescreve env vars)
        """
        self._settings = settings or get_settings()

        # Override settings com parâmetros explícitos
        self._classifier_model_path = (
            Path(classifier_model_path)
            if classifier_model_path
            else self._settings.classifier_model_path
        )
        self._classifier_model_type = (
            classifier_model_type or self._settings.classifier_model_type
        )
        self._classifier_device = (
            classifier_device or self._settings.classifier_device
        )
        self._classifier_fp8 = (
            classifier_fp8 if classifier_fp8 is not None else self._settings.classifier_fp8
        )

        # Extractor settings
        if extractor_backend:
            if isinstance(extractor_backend, str):
                extractor_backend = ExtractorBackend(extractor_backend)
            self._extractor_backend = extractor_backend
        else:
            self._extractor_backend = self._settings.extractor_backend

        self._extractor_device = extractor_device or self._settings.extractor_device

        # Lazy-loaded components
        self._classifier: ClassifierAdapter | None = None
        self._extractor: BaseExtractor | None = None

    @property
    def classifier(self) -> ClassifierAdapter:
        """Retorna o classificador (lazy-loaded)."""
        if self._classifier is None:
            self._classifier = ClassifierAdapter(
                model_path=self._classifier_model_path,
                model_type=self._classifier_model_type,
                device=self._classifier_device,
                fp8=self._classifier_fp8,
            )
        return self._classifier

    @property
    def extractor(self) -> BaseExtractor:
        """Retorna o extractor (lazy-loaded)."""
        if self._extractor is None:
            if self._extractor_backend == ExtractorBackend.QWEN_VL:
                self._extractor = QwenVLExtractor(
                    model_name=self._settings.extractor_model_qwen,
                    device=self._extractor_device,
                )
            elif self._extractor_backend == ExtractorBackend.GOT_OCR:
                self._extractor = GOTOCRExtractor(
                    model_name=self._settings.extractor_model_got,
                    device=self._extractor_device,
                )
            else:
                raise ValueError(f"Backend não suportado: {self._extractor_backend}")
        return self._extractor

    @property
    def classes(self) -> list[str]:
        """Lista de classes suportadas."""
        return self.classifier.classes

    def classify(self, image: str | Path | Image.Image) -> ClassificationResult:
        """
        Classifica uma imagem de documento.

        Args:
            image: Caminho da imagem ou PIL.Image

        Returns:
            ClassificationResult com tipo e confiança
        """
        return self.classifier.classify(image)

    def extract(
        self,
        image: str | Path | Image.Image,
        document_type: DocumentType | str,
    ) -> ExtractionResult:
        """
        Extrai dados de uma imagem de documento.

        Args:
            image: Caminho da imagem ou PIL.Image
            document_type: Tipo do documento (ou string)

        Returns:
            ExtractionResult com dados extraídos
        """
        if isinstance(document_type, str):
            document_type = DocumentType(document_type)

        return self.extractor.extract(image, document_type)

    def extract_generic(
        self,
        image: str | Path | Image.Image,
    ) -> GenericExtractionResult:
        """
        Extrai texto bruto de uma imagem (OCR puro).

        Args:
            image: Caminho da imagem ou PIL.Image

        Returns:
            GenericExtractionResult com texto extraído
        """
        return self.extractor.extract_generic(image)

    def extract_generic_from_pdf(
        self,
        pdf_path: str | Path,
    ) -> GenericExtractionResult:
        """
        Extrai texto bruto de um PDF (OCR puro, multi-page).

        Args:
            pdf_path: Caminho do arquivo PDF

        Returns:
            GenericExtractionResult com texto de todas as páginas
        """
        return self.extractor.extract_generic_from_pdf(pdf_path)

    def process(
        self,
        image: str | Path | Image.Image,
        extract: bool = True,
        min_confidence: float | None = None,
    ) -> PipelineResult:
        """
        Processa uma imagem: classifica e opcionalmente extrai dados.

        Args:
            image: Caminho da imagem ou PIL.Image
            extract: Se deve extrair dados após classificar
            min_confidence: Confiança mínima para classificação

        Returns:
            PipelineResult com classificação e extração
        """
        min_confidence = min_confidence or self._settings.min_confidence
        file_path = str(image) if isinstance(image, (str, Path)) else None

        try:
            # Classificação
            classification = self.classify(image)

            # Verifica confiança
            if classification.confidence < min_confidence:
                return PipelineResult(
                    file_path=file_path,
                    classification=classification,
                    extraction=None,
                    success=False,
                    error=f"Confiança ({classification.confidence:.1%}) abaixo do mínimo ({min_confidence:.1%})",
                )

            # Extração
            extraction = None
            if extract:
                extraction = self.extract(image, classification.document_type)

            return PipelineResult(
                file_path=file_path,
                classification=classification,
                extraction=extraction,
                success=True,
            )

        except Exception as e:
            return PipelineResult(
                file_path=file_path,
                classification=ClassificationResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                ),
                extraction=None,
                success=False,
                error=str(e),
            )

    def process_folder(
        self,
        folder: str | Path,
        extract: bool = True,
        min_confidence: float | None = None,
        extensions: list[str] | None = None,
    ) -> Iterator[PipelineResult]:
        """
        Processa todas as imagens de uma pasta.

        Args:
            folder: Caminho da pasta
            extract: Se deve extrair dados
            min_confidence: Confiança mínima
            extensions: Extensões de arquivo a processar

        Yields:
            PipelineResult para cada imagem
        """
        folder = Path(folder)
        extensions = extensions or [
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
            "*.heic", "*.heif", "*.avif",  # Formatos de smartphone
        ]

        # Padrões a ignorar
        ignore_patterns = ["_gt_", "_segmentation", "_mask", "_ocr"]

        images = []
        for ext in extensions:
            images.extend(folder.rglob(ext))

        # Filtra arquivos ignorados
        images = [
            img
            for img in images
            if not any(p in img.name for p in ignore_patterns)
        ]

        for img_path in images:
            yield self.process(
                img_path,
                extract=extract,
                min_confidence=min_confidence,
            )

    def unload_extractor(self) -> None:
        """Descarrega o extractor para liberar memória."""
        if self._extractor is not None:
            self._extractor.unload_model()
            self._extractor = None

    def warmup(self, load_classifier: bool = True, load_extractor: bool = True) -> None:
        """
        Carrega os modelos antecipadamente (warmup).

        Isso inclui download dos pesos se ainda não estiverem em cache.

        Args:
            load_classifier: Se deve carregar o classificador
            load_extractor: Se deve carregar o extractor
        """
        if load_classifier:
            # ClassifierAdapter carrega modelo no __init__
            _ = self.classifier
        if load_extractor:
            # Extractor usa lazy loading, precisa chamar load_model() explicitamente
            self.extractor.load_model()
