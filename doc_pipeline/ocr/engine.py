"""OCR Engine using PaddleOCR."""

import logging
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR wrapper for text extraction."""

    def __init__(
        self,
        lang: str = "pt",
        use_gpu: bool = True,
        gpu_id: int = 0,
        use_angle_cls: bool = True,
        show_log: bool = False,
    ):
        """
        Initialize OCR Engine.

        Args:
            lang: Language code (pt, en, ch, etc.)
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID
            use_angle_cls: Use angle classifier for rotated text
            show_log: Show PaddleOCR logs
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._ocr: PaddleOCR | None = None

    @property
    def ocr(self) -> PaddleOCR:
        """Lazy load PaddleOCR instance."""
        if self._ocr is None:
            logger.info(f"Loading PaddleOCR (lang={self.lang}, gpu={self.use_gpu})")
            # PaddleOCR 2.x API
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_gpu=self.use_gpu,
                gpu_mem=2000,  # Limit GPU memory to 2GB
                use_angle_cls=self.use_angle_cls,
                show_log=self.show_log,
            )
            logger.info("PaddleOCR loaded")
        return self._ocr

    def extract_text(
        self,
        image: Image.Image | str | Path,
        preserve_layout: bool = False,
    ) -> tuple[str, float]:
        """
        Extract text from image.

        Args:
            image: PIL Image or path to image file
            preserve_layout: Try to preserve original text layout

        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = str(image)

        # Run OCR
        result = self.ocr.ocr(img_array, cls=self.use_angle_cls)

        if not result or not result[0]:
            return "", 0.0

        # Extract text and confidence
        lines = []
        confidences = []

        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]
                confidence = line[1][1]
                lines.append(text)
                confidences.append(confidence)

        # Join text
        if preserve_layout:
            # TODO: Use bounding boxes to preserve layout
            text = "\n".join(lines)
        else:
            text = "\n".join(lines)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_confidence

    def extract_with_boxes(
        self,
        image: Image.Image | str | Path,
    ) -> list[dict]:
        """
        Extract text with bounding boxes.

        Args:
            image: PIL Image or path to image file

        Returns:
            List of dicts with keys: text, confidence, bbox (x1,y1,x2,y2)
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = str(image)

        result = self.ocr.ocr(img_array, cls=self.use_angle_cls)

        if not result or not result[0]:
            return []

        extractions = []
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = line[1][0]
                confidence = line[1][1]

                # Convert bbox to simple format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                extractions.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": {
                        "x1": min(x_coords),
                        "y1": min(y_coords),
                        "x2": max(x_coords),
                        "y2": max(y_coords),
                    },
                })

        return extractions

    def warmup(self):
        """Warmup the model by running a dummy inference."""
        logger.info("Warming up PaddleOCR...")
        # Create a small dummy image
        dummy_img = Image.new("RGB", (100, 50), color="white")
        self.extract_text(dummy_img)
        logger.info("PaddleOCR warmup complete")
