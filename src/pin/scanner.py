"""High-level score-sheet scanner."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pin.classification import boxed_throw_scores, classify_circles, classify_dashes
from pin.detection import (
    OnnxDetector,
    crop_detections,
    detect_bordered_diagrams,
    sort_detections,
)
from pin.preprocess import rectify_sheet
from pin.types import Classification, Detection, SheetResult, ThrowResult, total

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/pin_diagram.onnx"


class Scanner:
    """Detect and classify pin diagrams while reusing one ONNX model."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._detector: OnnxDetector | None = None

    @property
    def detector(self) -> OnnxDetector:
        if self._detector is None:
            self._detector = OnnxDetector(self.model_path)
        return self._detector

    def detect_image(
        self, image: np.ndarray, confidence: float = 0.25
    ) -> tuple[np.ndarray, list[Detection]]:
        """Rectify an image and detect every diagram."""
        rectified = rectify_sheet(image)
        detections = detect_bordered_diagrams(rectified)
        if len(detections) < 10:
            detections = self.detector.detect(rectified, confidence)
        return rectified, sort_detections(detections)

    def detect_sheet(
        self, image_path: Path | str, confidence: float = 0.25
    ) -> tuple[np.ndarray, list[Detection]]:
        """Load a sheet and detect every diagram."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.detect_image(image, confidence)

    def scan_image(self, image: np.ndarray, confidence: float = 0.25) -> SheetResult:
        """Detect and classify all diagrams in an image."""
        rectified, detections = self.detect_image(image, confidence)
        crops = crop_detections(rectified, detections)
        if not crops:
            return SheetResult()

        circle_results = [classify_circles(crop) for crop in crops]
        boxed_layout = (
            sum(result is not None for result in circle_results) > len(crops) / 2
        )
        if boxed_layout:
            classifications = [
                result
                if result is not None
                else Classification((None,) * 9, 0.0, "unknown")
                for result in circle_results
            ]
        else:
            classifications = [
                result
                if (result := classify_dashes(crop)) is not None
                else Classification((None,) * 9, 0.0, "unknown")
                for crop in crops
            ]

        entries = [
            (detection.column, detection.row, classification.pins)
            for detection, classification in zip(detections, classifications)
        ]
        scores = (
            boxed_throw_scores(entries)
            if boxed_layout
            else [total(pins) for _, _, pins in entries]
        )
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
        return SheetResult(
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

    def scan_sheet(
        self, image_path: Path | str, confidence: float = 0.25
    ) -> SheetResult:
        """Load and scan one score sheet."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.scan_image(image, confidence)
