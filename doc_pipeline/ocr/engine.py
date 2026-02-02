"""OCR Engine using EasyOCR."""

import logging
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """EasyOCR wrapper for text extraction with good Portuguese support."""

    def __init__(
        self,
        lang: str = "pt",
        use_gpu: bool = False,
        gpu_id: int = 0,
        show_log: bool = False,
    ):
        """
        Initialize OCR Engine.

        Args:
            lang: Language code (pt, en, etc.)
            use_gpu: Whether to use GPU
            gpu_id: GPU device ID (not used by EasyOCR directly)
            show_log: Show verbose logs
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.show_log = show_log
        self._reader: easyocr.Reader | None = None

    @property
    def reader(self) -> easyocr.Reader:
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            # Map common language codes to EasyOCR format
            lang_map = {
                "pt": ["pt"],
                "en": ["en"],
                "latin": ["pt", "en"],  # Use both for latin
                "es": ["es"],
                "fr": ["fr"],
                "de": ["de"],
                "it": ["it"],
            }

            languages = lang_map.get(self.lang, [self.lang])

            logger.info(f"Loading EasyOCR (languages={languages}, gpu={self.use_gpu})")
            self._reader = easyocr.Reader(
                languages,
                gpu=self.use_gpu,
                verbose=self.show_log,
            )
            logger.info("EasyOCR loaded")
        return self._reader

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
        # EasyOCR returns list of (bbox, text, confidence)
        result = self.reader.readtext(img_array)

        if not result:
            return "", 0.0

        # Extract text and confidence
        lines = []
        confidences = []

        for detection in result:
            if len(detection) >= 3:
                text = detection[1]
                confidence = detection[2]
                lines.append(text)
                confidences.append(confidence)

        # Join text
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

        result = self.reader.readtext(img_array)

        if not result:
            return []

        extractions = []
        for detection in result:
            if len(detection) >= 3:
                bbox = detection[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                text = detection[1]
                confidence = detection[2]

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
        logger.info("Warming up EasyOCR...")
        # Create a small dummy image
        dummy_img = Image.new("RGB", (100, 50), color="white")
        self.extract_text(dummy_img)
        logger.info("EasyOCR warmup complete")
