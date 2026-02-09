"""
Image orientation detection and correction.

Uses a combination of:
- EXIF metadata
- EasyOCR text box direction (for 90°/270° detection)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import ExifTags, Image

from doc_pipeline.observability import get_logger

logger = get_logger(__name__)

# Minimum avg width/height ratio to consider text as horizontal
_HORIZONTAL_THRESHOLD = 1.5
# Minimum number of text boxes needed for reliable detection
_MIN_TEXTBOXES = 3


class RotationAngle(int, Enum):
    """Possible rotation angles."""

    NONE = 0
    CW_90 = 90  # Clockwise 90°
    CW_180 = 180  # Upside down
    CCW_90 = 270  # Counter-clockwise 90° (or CW 270°)


@dataclass
class OrientationResult:
    """Result of orientation correction."""

    image: Image.Image
    was_corrected: bool
    rotation_applied: RotationAngle
    correction_method: str | None  # "exif", "text", "exif+text"
    confidence: float | None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "was_corrected": self.was_corrected,
            "rotation_applied": self.rotation_applied.value if self.was_corrected else 0,
            "correction_method": self.correction_method,
            "confidence": self.confidence,
        }


class OrientationCorrector:
    """
    Corrects image orientation using EXIF data and text detection.

    Strategy:
    1. Apply EXIF orientation (handles camera rotation metadata)
    2. EasyOCR text box direction for 90°/270° detection (~100ms)
    """

    def __init__(
        self,
        use_text_detection: bool = True,
        device: str = "cuda:0",
        confidence_threshold: float = 0.3,
        ocr_engine=None,
    ):
        self.use_text_detection = use_text_detection
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._ocr_engine = ocr_engine

    def set_ocr_engine(self, ocr_engine) -> None:
        """Set OCR engine for text box direction detection."""
        self._ocr_engine = ocr_engine

    def warmup(self) -> None:
        """Pre-load models and warm up CUDA kernels."""
        logger.info("orientation_warmup_start")
        # EasyOCR warms up via its own warmup method; nothing extra needed here
        logger.info("orientation_warmup_complete")

    def correct(self, image: Image.Image) -> OrientationResult:
        """Detect and correct image orientation."""
        logger.info("orientation_correction_starting", image_size=image.size)

        total_rotation = RotationAngle.NONE
        correction_method = None
        confidence = None

        # Step 1: Apply EXIF orientation
        exif_corrected, exif_rotation = self._apply_exif_orientation(image)
        if exif_rotation != RotationAngle.NONE:
            image = exif_corrected
            total_rotation = exif_rotation
            correction_method = "exif"
            logger.info("exif_orientation_applied", rotation=exif_rotation.value)

        # Step 2: Text-based orientation detection
        if self.use_text_detection:
            text_corrected, text_rotation, text_confidence = self._detect_text_orientation(image)
            if text_rotation != RotationAngle.NONE:
                image = text_corrected
                total_rotation = RotationAngle((total_rotation.value + text_rotation.value) % 360)
                correction_method = (
                    correction_method + "+text" if correction_method else "text"
                )
                confidence = text_confidence
                logger.info(
                    "text_orientation_corrected",
                    rotation=text_rotation.value,
                    confidence=text_confidence,
                )

        was_corrected = total_rotation != RotationAngle.NONE

        if was_corrected:
            logger.info(
                "orientation_correction_complete",
                total_rotation=total_rotation.value,
                method=correction_method,
            )

        return OrientationResult(
            image=image,
            was_corrected=was_corrected,
            rotation_applied=total_rotation,
            correction_method=correction_method if was_corrected else None,
            confidence=confidence,
        )

    def _apply_exif_orientation(self, image: Image.Image) -> tuple[Image.Image, RotationAngle]:
        """Apply EXIF orientation tag to correct camera rotation."""
        try:
            exif = image.getexif()
            if not exif:
                return image, RotationAngle.NONE

            orientation_tag = None
            for tag, name in ExifTags.TAGS.items():
                if name == "Orientation":
                    orientation_tag = tag
                    break

            if orientation_tag is None or orientation_tag not in exif:
                return image, RotationAngle.NONE

            orientation = exif[orientation_tag]

            if orientation == 3:
                return image.rotate(180, expand=True), RotationAngle.CW_180
            elif orientation == 6:
                return image.rotate(270, expand=True), RotationAngle.CW_90
            elif orientation == 8:
                return image.rotate(90, expand=True), RotationAngle.CCW_90

            return image, RotationAngle.NONE

        except Exception as e:
            logger.warning("exif_orientation_error", error=str(e))
            return image, RotationAngle.NONE

    def _detect_text_orientation(
        self, image: Image.Image
    ) -> tuple[Image.Image, RotationAngle, float | None]:
        """
        Orientation detection via EasyOCR text boxes for 90°/270°
        (avg box width/height ratio).
        """
        try:
            if self._ocr_engine is None:
                return image, RotationAngle.NONE, None

            img_array = np.array(image)
            result_90 = self._detect_90_with_textboxes(img_array)

            if result_90 is None:
                return image, RotationAngle.NONE, None

            rotation, pil_deg, conf = result_90
            corrected = image.rotate(pil_deg, expand=True)
            return corrected, rotation, conf

        except Exception as e:
            logger.warning(
                "text_orientation_detection_error", error=str(e), error_type=type(e).__name__
            )
            return image, RotationAngle.NONE, None

    # ------------------------------------------------------------------
    # EasyOCR text-box direction detection
    # ------------------------------------------------------------------

    def _avg_box_wh_ratio(self, img_array: np.ndarray) -> tuple[float, int]:
        """
        Run EasyOCR detect and return the average width/height ratio of text boxes.

        Returns:
            (avg_ratio, total_boxes)
            - avg_ratio > 2: text is clearly horizontal
            - avg_ratio < 1: text is clearly vertical
        """
        horizontal_list, free_list = self._ocr_engine.reader.detect(img_array)
        boxes = horizontal_list[0] if horizontal_list else []

        if not boxes:
            return 0.0, 0

        ratios = [(b[1] - b[0]) / max(b[3] - b[2], 1) for b in boxes]
        return sum(ratios) / len(ratios), len(boxes)

    def _detect_90_with_textboxes(
        self, img_array: np.ndarray
    ) -> tuple[RotationAngle, int, float] | None:
        """
        Detect 90° rotation using text box width/height ratios.

        If text boxes are mostly vertical (avg w/h < threshold), tries both
        90° CW and 90° CCW rotations and picks the best one using OCR confidence.

        Returns:
            (RotationAngle, pil_degrees, confidence) or None if no rotation needed.
        """
        avg_ratio, total = self._avg_box_wh_ratio(img_array)

        logger.info(
            "textbox_direction_check",
            avg_wh_ratio=round(avg_ratio, 2),
            total_boxes=total,
            threshold=_HORIZONTAL_THRESHOLD,
        )

        if total < _MIN_TEXTBOXES:
            logger.info("textbox_too_few_boxes", total=total)
            return None

        if avg_ratio >= _HORIZONTAL_THRESHOLD:
            # Text is horizontal → not rotated 90°
            return None

        # Text appears vertical → try both 90° rotations
        img_cw = np.ascontiguousarray(np.rot90(img_array, k=3))   # 90° CW
        img_ccw = np.ascontiguousarray(np.rot90(img_array, k=1))  # 90° CCW

        ratio_cw, _ = self._avg_box_wh_ratio(img_cw)
        ratio_ccw, _ = self._avg_box_wh_ratio(img_ccw)

        logger.info(
            "textbox_90_candidates",
            ratio_cw=round(ratio_cw, 2),
            ratio_ccw=round(ratio_ccw, 2),
        )

        # If both rotations give horizontal text, use OCR confidence to decide
        if ratio_cw >= _HORIZONTAL_THRESHOLD and ratio_ccw >= _HORIZONTAL_THRESHOLD:
            return self._tiebreak_with_ocr(img_cw, img_ccw)

        # One clearly better
        if ratio_cw >= _HORIZONTAL_THRESHOLD and ratio_cw > ratio_ccw:
            logger.info("textbox_90_selected", direction="CW", ratio=round(ratio_cw, 2))
            return (RotationAngle.CW_90, 270, ratio_cw)

        if ratio_ccw >= _HORIZONTAL_THRESHOLD:
            logger.info("textbox_90_selected", direction="CCW", ratio=round(ratio_ccw, 2))
            return (RotationAngle.CCW_90, 90, ratio_ccw)

        # Neither rotation makes text horizontal
        logger.info("textbox_90_inconclusive", ratio_cw=round(ratio_cw, 2), ratio_ccw=round(ratio_ccw, 2))
        return None

    def _tiebreak_with_ocr(
        self, img_cw: np.ndarray, img_ccw: np.ndarray
    ) -> tuple[RotationAngle, int, float] | None:
        """
        Break tie between 90° CW and CCW using EasyOCR recognition confidence.
        The correct orientation yields higher average OCR confidence.
        """
        results_cw = self._ocr_engine.reader.readtext(img_cw)
        results_ccw = self._ocr_engine.reader.readtext(img_ccw)

        conf_cw = (
            sum(r[2] for r in results_cw) / len(results_cw)
            if results_cw
            else 0.0
        )
        conf_ccw = (
            sum(r[2] for r in results_ccw) / len(results_ccw)
            if results_ccw
            else 0.0
        )

        logger.info(
            "textbox_90_tiebreak_ocr",
            conf_cw=round(conf_cw, 3),
            texts_cw=len(results_cw),
            conf_ccw=round(conf_ccw, 3),
            texts_ccw=len(results_ccw),
        )

        if conf_cw > conf_ccw:
            return (RotationAngle.CW_90, 270, conf_cw)
        elif conf_ccw > conf_cw:
            return (RotationAngle.CCW_90, 90, conf_ccw)

        return None
