"""High-level score-sheet scanner."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pin.classification import CropClassifier, boxed_throw_scores
from pin.detection import (
    DEFAULT_CONFIDENCE,
    YoloDetector,
    crop_detections,
    sort_detections,
)
from pin.types import Classification, Detection, SheetResult, ThrowResult

MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
DEFAULT_EXTRACTOR_PATH = MODEL_DIRECTORY / "field_extractor.onnx"
DEFAULT_CLASSIFIER_PATH = MODEL_DIRECTORY / "pin_classifier.onnx"


class Scanner:
    """Detect and classify diagrams from bordered score sheets."""

    def __init__(
        self,
        extractor_path: Path | str = DEFAULT_EXTRACTOR_PATH,
        classifier_path: Path | str = DEFAULT_CLASSIFIER_PATH,
    ) -> None:
        self.extractor_path = Path(extractor_path)
        self.classifier_path = Path(classifier_path)
        self._detector: YoloDetector | None = None
        self._classifier: CropClassifier | None = None

    @property
    def detector(self) -> YoloDetector:
        if self._detector is None:
            self._detector = YoloDetector(self.extractor_path)
        return self._detector

    @property
    def classifier(self) -> CropClassifier:
        if self._classifier is None:
            self._classifier = CropClassifier(str(self.classifier_path))
        return self._classifier

    def detect_image(
        self, image: np.ndarray, confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[np.ndarray, list[Detection]]:
        """Detect every pin field in an image."""
        detections = self.detector.detect(image, confidence)
        return image, sort_detections(detections)

    def detect_sheet(
        self, image_path: Path | str, confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[np.ndarray, list[Detection]]:
        """Load a sheet and detect every diagram."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.detect_image(image, confidence)

    def classify_detections(
        self, image: np.ndarray, detections: list[Detection]
    ) -> tuple[SheetResult, list[np.ndarray], list[Classification]]:
        """Classify detected crops and return the intermediate results."""
        crops = crop_detections(image, detections)
        if not crops:
            return SheetResult(), [], []
        classifications = [self.classifier.classify(crop) for crop in crops]
        entries = [
            (detection.column, detection.row, classification.pins)
            for detection, classification in zip(detections, classifications)
        ]
        scores = boxed_throw_scores(entries)
        throws = [
            ThrowResult(
                detection.column,
                detection.row,
                classification.pins,
                score,
                detection.confidence,
                classification.confidence,
            )
            for detection, classification, score in zip(
                detections, classifications, scores
            )
        ]
        columns = {detection.column for detection in detections}
        result = SheetResult(
            throws,
            len(columns),
            max(
                (
                    sum(detection.column == column for detection in detections)
                    for column in columns
                ),
                default=0,
            ),
        )
        return result, crops, classifications

    def scan_image(
        self, image: np.ndarray, confidence: float = DEFAULT_CONFIDENCE
    ) -> SheetResult:
        """Run the sheet detector and crop classifier in sequence."""
        detected_image, detections = self.detect_image(image, confidence)
        result, _, _ = self.classify_detections(detected_image, detections)
        return result

    def scan_sheet(
        self, image_path: Path | str, confidence: float = DEFAULT_CONFIDENCE
    ) -> SheetResult:
        """Load and scan one score sheet."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.scan_image(image, confidence)
