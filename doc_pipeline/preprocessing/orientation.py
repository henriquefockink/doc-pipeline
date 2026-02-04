"""
Image orientation detection and correction.

Uses a combination of EXIF data and text orientation detection
to automatically correct rotated images.
"""

from dataclasses import dataclass
from enum import Enum

from PIL import Image, ExifTags

from doc_pipeline.observability import get_logger

logger = get_logger(__name__)


class RotationAngle(int, Enum):
    """Possible rotation angles."""
    NONE = 0
    CW_90 = 90      # Clockwise 90°
    CW_180 = 180    # Upside down
    CCW_90 = 270    # Counter-clockwise 90° (or CW 270°)


@dataclass
class OrientationResult:
    """Result of orientation correction."""

    image: Image.Image
    was_corrected: bool
    rotation_applied: RotationAngle
    correction_method: str | None  # "exif", "text_detection", or None
    confidence: float | None  # Confidence of text-based detection

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
    1. First apply EXIF orientation (handles camera rotation metadata)
    2. Then detect text orientation using OCR and correct if needed
    """

    def __init__(self, use_text_detection: bool = True):
        """
        Initialize the orientation corrector.

        Args:
            use_text_detection: Whether to use OCR-based text detection
                               for orientation correction. Requires easyocr.
        """
        self.use_text_detection = use_text_detection
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Lazy load OCR engine."""
        if self._ocr_engine is None:
            from doc_pipeline.ocr import OCREngine
            self._ocr_engine = OCREngine(lang="latin", use_gpu=True)
        return self._ocr_engine

    def correct(self, image: Image.Image) -> OrientationResult:
        """
        Detect and correct image orientation.

        Args:
            image: PIL Image to correct

        Returns:
            OrientationResult with corrected image and metadata
        """
        logger.info("orientation_correction_starting", image_size=image.size)

        original_image = image
        total_rotation = RotationAngle.NONE
        correction_method = None
        confidence = None

        # Step 1: Apply EXIF orientation
        exif_corrected, exif_rotation = self._apply_exif_orientation(image)
        if exif_rotation != RotationAngle.NONE:
            image = exif_corrected
            total_rotation = exif_rotation
            correction_method = "exif"
            logger.info(
                "exif_orientation_applied",
                rotation=exif_rotation.value,
            )

        # Step 2: Text-based orientation detection
        if self.use_text_detection:
            text_corrected, text_rotation, text_confidence = self._detect_text_orientation(image)
            if text_rotation != RotationAngle.NONE:
                image = text_corrected
                # Combine rotations
                total_rotation = RotationAngle((total_rotation.value + text_rotation.value) % 360)
                correction_method = "text_detection" if correction_method is None else "exif+text_detection"
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
        """
        Apply EXIF orientation tag to correct camera rotation.

        Returns:
            Tuple of (corrected image, rotation applied)
        """
        try:
            exif = image.getexif()
            if not exif:
                return image, RotationAngle.NONE

            # Find orientation tag
            orientation_tag = None
            for tag, name in ExifTags.TAGS.items():
                if name == "Orientation":
                    orientation_tag = tag
                    break

            if orientation_tag is None or orientation_tag not in exif:
                return image, RotationAngle.NONE

            orientation = exif[orientation_tag]

            # EXIF orientation values:
            # 1: Normal
            # 3: Rotated 180°
            # 6: Rotated 90° CW
            # 8: Rotated 90° CCW

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
        Detect text orientation using OCR and correct if needed.

        Strategy:
        - Run OCR at 0°, 90°, 180°, 270°
        - The orientation with highest average confidence is likely correct
        - Only correct if confidence difference is significant

        Returns:
            Tuple of (corrected image, rotation applied, confidence)
        """
        try:
            import numpy as np

            ocr = self._get_ocr_engine()

            # Test all 4 orientations
            rotations = [
                (RotationAngle.NONE, 0),
                (RotationAngle.CW_90, 270),   # PIL rotate is CCW, so 270 = CW 90
                (RotationAngle.CW_180, 180),
                (RotationAngle.CCW_90, 90),   # PIL rotate 90 = CCW 90
            ]

            results = []

            for rotation_enum, pil_rotation in rotations:
                # Rotate image
                if pil_rotation == 0:
                    rotated = image
                else:
                    rotated = image.rotate(pil_rotation, expand=True)

                # Convert to numpy for OCR
                img_array = np.array(rotated)

                # Run OCR (readtext returns list of (bbox, text, confidence))
                ocr_results = ocr.reader.readtext(img_array)

                if not ocr_results:
                    results.append((rotation_enum, pil_rotation, 0.0, 0))
                    continue

                # Calculate average confidence weighted by text length
                total_weight = 0
                weighted_confidence = 0

                for _, text, conf in ocr_results:
                    weight = len(text)
                    weighted_confidence += conf * weight
                    total_weight += weight

                avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
                results.append((rotation_enum, pil_rotation, avg_confidence, len(ocr_results)))

            # Log all results for debugging
            for rot, _, conf, count in results:
                logger.info(
                    "orientation_test_result",
                    rotation=rot.value,
                    confidence=round(conf, 4),
                    text_count=count,
                )

            # Find best orientation using a combined score
            # Weight both confidence and text count
            max_count = max(r[3] for r in results) or 1

            def calc_score(r):
                rot, pil_rot, conf, count = r
                # Normalize text count (max becomes 1.0)
                norm_count = count / max_count
                # Combined score: 60% confidence + 40% text count
                return conf * 0.6 + norm_count * 0.4

            # Calculate scores for all results
            scored_results = [(r, calc_score(r)) for r in results]
            scored_results.sort(key=lambda x: x[1], reverse=True)

            best_result, best_score = scored_results[0]
            best_rotation, best_pil_rotation, best_confidence, best_text_count = best_result

            # Also get original (0°) result for comparison
            original_result = next(r for r in results if r[0] == RotationAngle.NONE)
            original_score = calc_score(original_result)

            logger.info(
                "orientation_scores",
                best_rotation=best_rotation.value,
                best_score=round(best_score, 4),
                original_score=round(original_score, 4),
            )

            # Only correct if:
            # 1. Best is not original (0°)
            # 2. Best score is meaningfully better than original
            original_confidence = original_result[2]
            min_score_improvement = 0.02  # At least 2% score improvement needed

            should_correct = (
                best_rotation != RotationAngle.NONE
                and best_confidence >= 0.15  # Minimum confidence threshold
                and (best_score - original_score) >= min_score_improvement
            )

            if should_correct:
                corrected = image.rotate(best_pil_rotation, expand=True)
                logger.info(
                    "orientation_correction_applied",
                    best_rotation=best_rotation.value,
                    best_score=round(best_score, 4),
                    original_score=round(original_score, 4),
                    improvement=round(best_score - original_score, 4),
                )
                return corrected, best_rotation, best_confidence

            logger.info(
                "orientation_no_correction_needed",
                best_rotation=best_rotation.value,
                best_score=round(best_score, 4),
                original_score=round(original_score, 4),
            )
            return image, RotationAngle.NONE, original_confidence

        except Exception as e:
            logger.warning("text_orientation_detection_error", error=str(e), error_type=type(e).__name__)
            import traceback
            logger.warning("text_orientation_detection_traceback", tb=traceback.format_exc())
            return image, RotationAngle.NONE, None
