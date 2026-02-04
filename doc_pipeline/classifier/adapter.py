"""
Adapter para o classificador de documentos.

O código de inferência foi copiado do repo doc-classifier para evitar
dependência externa. Ver docs/DEPENDENCIA_DOC_CLASSIFIER.md para contexto.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ..schemas import ClassificationResult, DocumentType

if TYPE_CHECKING:
    from .classificar import ClassificadorDocumentos


class ClassifierAdapter:
    """Wrapper para ClassificadorDocumentos."""

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "efficientnet_b0",
        device: str | None = None,
        fp8: bool = False,
    ):
        """
        Inicializa o adapter.

        Args:
            model_path: Caminho para o modelo .pth
            model_type: Tipo do modelo (efficientnet_b0, b2, b4)
            device: Device para inferência (cuda:0, cpu, etc)
            fp8: Usar quantização FP8
        """
        from .classificar import ClassificadorDocumentos

        self._classifier = ClassificadorDocumentos(
            modelo_path=str(model_path),
            modelo_tipo=model_type,
            device=device,
            fp8=fp8,
        )

    @property
    def classes(self) -> list[str]:
        """Lista de classes suportadas."""
        return self._classifier.classes

    @property
    def device(self) -> str:
        """Device usado pelo modelo."""
        return self._classifier.device

    def classify(self, image: str | Path | Image.Image) -> ClassificationResult:
        """
        Classifica uma imagem.

        Args:
            image: Caminho da imagem ou PIL.Image

        Returns:
            ClassificationResult com tipo, confiança e probabilidades
        """
        result = self._classifier.classificar(image)

        return ClassificationResult(
            document_type=DocumentType(result["classe"]),
            confidence=result["confianca"],
        )

    def classify_batch(
        self, images: list[str | Path | Image.Image]
    ) -> list[ClassificationResult]:
        """Classifica múltiplas imagens."""
        return [self.classify(img) for img in images]
