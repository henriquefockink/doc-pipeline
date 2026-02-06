"""
Image orientation detection and correction.

Uses a combination of EXIF data and docTR page orientation classification
to automatically correct rotated images.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import ExifTags, Image

from doc_pipeline.observability import get_logger

logger = get_logger(__name__)


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
    correction_method: str | None  # "exif", "doctr_classification", or None
    confidence: float | None  # Confidence of docTR-based detection

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "was_corrected": self.was_corrected,
            "rotation_applied": self.rotation_applied.value if self.was_corrected else 0,
            "correction_method": self.correction_method,
            "confidence": self.confidence,
        }


# Mapping from docTR angle output to RotationAngle.
# docTR returns the angle the image IS rotated by, so we need to
# apply the same rotation to correct it (PIL rotate is CCW).
# docTR angle -> (RotationAngle, PIL rotation degrees)
_DOCTR_ANGLE_MAP: dict[int, tuple[RotationAngle, int]] = {
    0: (RotationAngle.NONE, 0),
    -90: (RotationAngle.CW_90, 270),  # Image is rotated 90° CW → rotate 270° CCW to fix
    180: (RotationAngle.CW_180, 180),
    90: (RotationAngle.CCW_90, 90),  # Image is rotated 90° CCW → rotate 90° CCW to fix
}


class OrientationCorrector:
    """
    Corrects image orientation using EXIF data and docTR classification.

    Strategy:
    1. First apply EXIF orientation (handles camera rotation metadata)
    2. Then classify page orientation using docTR MobileNetV3 (~50ms)
    """

    def __init__(
        self,
        use_text_detection: bool = True,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        use_torch_compile: bool = False,
        ocr_engine=None,  # kept for backward compatibility, ignored
    ):
        """
        Initialize the orientation corrector.

        Args:
            use_text_detection: Whether to use docTR-based orientation detection.
            device: Torch device for the orientation model (e.g. "cuda:0", "cpu").
            confidence_threshold: Minimum confidence to apply correction (0-1).
            use_torch_compile: Whether to apply torch.compile to the model.
            ocr_engine: Ignored. Kept for backward compatibility.
        """
        self.use_text_detection = use_text_detection
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._use_torch_compile = use_torch_compile
        self._predictor = None

    def _get_predictor(self):
        """Get or lazy-load the docTR page orientation predictor."""
        if self._predictor is None:
            import torch
            from doctr.models import page_orientation_predictor

            logger.info(
                "loading_orientation_predictor",
                device=self._device,
                torch_compile=self._use_torch_compile,
            )

            self._predictor = page_orientation_predictor(
                arch="mobilenet_v3_small_page_orientation",
                pretrained=True,
            )

            # Move model to device
            device = torch.device(self._device)
            self._predictor.model = self._predictor.model.to(device)

            if self._use_torch_compile:
                self._predictor.model = torch.compile(self._predictor.model)

            self._predictor.model.eval()

            logger.info("orientation_predictor_loaded", device=self._device)

        return self._predictor

    def warmup(self) -> None:
        """Run a dummy inference to pre-load the model and warm up CUDA kernels."""
        logger.info("orientation_warmup_start")
        predictor = self._get_predictor()
        dummy = (255 * np.random.rand(512, 512, 3)).astype(np.uint8)
        predictor([dummy])
        logger.info("orientation_warmup_complete")

    def correct(self, image: Image.Image) -> OrientationResult:
        """
        Detect and correct image orientation.

        Args:
            image: PIL Image to correct

        Returns:
            OrientationResult with corrected image and metadata
        """
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
            logger.info(
                "exif_orientation_applied",
                rotation=exif_rotation.value,
            )

        # Step 2: docTR-based orientation classification
        if self.use_text_detection:
            text_corrected, text_rotation, text_confidence = self._detect_text_orientation(image)
            if text_rotation != RotationAngle.NONE:
                image = text_corrected
                # Combine rotations
                total_rotation = RotationAngle((total_rotation.value + text_rotation.value) % 360)
                correction_method = (
                    "doctr_classification"
                    if correction_method is None
                    else "exif+doctr_classification"
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
        Detect page orientation using docTR MobileNetV3 classifier.

        Single forward pass (~50ms on GPU) instead of running OCR 4x.

        Returns:
            Tuple of (corrected image, rotation applied, confidence)
        """
        try:
            predictor = self._get_predictor()

            # Convert PIL image to numpy array (RGB, uint8)
            img_array = np.array(image)

            # docTR predictor expects list of numpy arrays
            # Returns: [class_idxs, angles, confidences]
            _class_idxs, angles, confidences = predictor([img_array])

            predicted_angle = angles[0]  # one of: 0, -90, 180, 90
            confidence = confidences[0]  # float 0-1

            logger.info(
                "doctr_orientation_result",
                predicted_angle=predicted_angle,
                confidence=round(confidence, 4),
                threshold=self._confidence_threshold,
            )

            # Map docTR angle to RotationAngle
            rotation_angle, pil_rotation = _DOCTR_ANGLE_MAP.get(
                predicted_angle, (RotationAngle.NONE, 0)
            )

            # Only correct if confidence exceeds threshold and rotation is needed
            if rotation_angle != RotationAngle.NONE and confidence >= self._confidence_threshold:
                corrected = image.rotate(pil_rotation, expand=True)
                logger.info(
                    "orientation_correction_applied",
                    rotation=rotation_angle.value,
                    confidence=round(confidence, 4),
                )
                return corrected, rotation_angle, confidence

            return image, RotationAngle.NONE, confidence

        except Exception as e:
            logger.warning(
                "text_orientation_detection_error", error=str(e), error_type=type(e).__name__
            )
            import traceback

            logger.warning("text_orientation_detection_traceback", tb=traceback.format_exc())
            return image, RotationAngle.NONE, None
