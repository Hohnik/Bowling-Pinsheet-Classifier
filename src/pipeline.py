"""Full pipeline: preprocess → detect → classify → OCR-validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

from symbols import boxed_throw_scores, classify_circle_pins, classify_dash_pins
from detect import (
    YOLOModel,
    Detection,
    crop_detections,
    detect_bordered_diagrams,
    detect_pin_diagrams_classical,
    detect_pin_diagrams_yolo,
    load_model,
    sort_detections,
)
from ocr import cross_validate
from preprocess import rectify_sheet

# ── Model caches (keyed by resolved path string) ──────────────────────────
# Loading YOLO or the CNN from disk takes ~100 ms per call.  When scanning
# multiple sheets in sequence the same weights file is used every time —
# cache the loaded objects so only the first call pays the I/O cost.


@lru_cache(maxsize=4)
def _cached_detector(path_str: str) -> YOLOModel:
    return load_model(Path(path_str))


@lru_cache(maxsize=4)
def _cached_classifier(path_str: str) -> tuple[Any, Any]:
    from classify import load_classifier

    return load_classifier(Path(path_str))


@dataclass
class ThrowResult:
    """Result for a single throw (one pin diagram)."""

    column: int
    row: int
    score: int
    pins_down: list[int] = field(default_factory=list)
    confidence: float = 0.0
    classification_confidence: float = 0.0
    ocr_mismatch: bool = False


@dataclass
class SheetResult:
    """Result for an entire score sheet."""

    throws: list[ThrowResult] = field(default_factory=list)
    columns: int = 0
    rows_per_column: int = 0

    @property
    def total_pins(self) -> int:
        return sum(t.score for t in self.throws)


DEFAULT_DETECTOR_PATH = Path("models/pin_diagram.pt")
DEFAULT_CLASSIFIER_PATH = Path("models/pin_classifier.pt")


def detect_sheet(
    image_path: Path,
    model_path: Path | None = None,
    confidence: float = 0.25,
) -> tuple[Any, list[Detection]]:
    """Load, rectify, and detect all pin diagrams in a sheet image."""
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    rectified = rectify_sheet(raw)

    bordered = detect_bordered_diagrams(rectified)
    if len(bordered) >= 10:
        return rectified, sort_detections(bordered)

    detector_path = model_path or DEFAULT_DETECTOR_PATH
    yolo_detections: list[Detection] = []
    if detector_path.exists():
        yolo = _cached_detector(str(detector_path.resolve()))
        yolo_detections = detect_pin_diagrams_yolo(yolo, rectified, confidence)
        if len(yolo_detections) >= 10:
            return rectified, sort_detections(yolo_detections)

    classical = detect_pin_diagrams_classical(rectified)
    return rectified, sort_detections(
        max((bordered, yolo_detections, classical), key=len)
    )


def process_sheet(
    image_path: Path,
    model_path: Path | None = None,
    classifier_path: Path | None = None,
    confidence: float = 0.25,
    use_ocr: bool = False,
) -> SheetResult:
    """Full pipeline: load → preprocess → detect → classify → OCR-validate.

    ``use_ocr=False`` (default) skips Tesseract entirely (~12× faster).
    Pass ``use_ocr=True`` to cross-validate CNN scores against printed digits.
    """
    classifier_path = classifier_path or DEFAULT_CLASSIFIER_PATH

    rectified, sorted_dets = detect_sheet(image_path, model_path, confidence)
    crops = crop_detections(rectified, sorted_dets)
    if not crops:
        return SheetResult()

    # Boxed circle diagrams have an exact geometric solution and need no CNN.
    circle_results = [classify_circle_pins(crop) for crop in crops]
    boxed_layout = all(result is not None for result in circle_results)
    if boxed_layout:
        classifications = [result for result in circle_results if result is not None]
    else:
        dash_results = [classify_dash_pins(crop) for crop in crops]
        if all(result is not None for result in dash_results):
            classifications = [result for result in dash_results if result is not None]
        else:
            from classify import classify_pins_batch

            if not classifier_path.exists():
                raise FileNotFoundError(
                    f"Classifier weights not found at {classifier_path}. "
                    "Train a model first (see `pinsheet-scanner train`)."
                )
            cnn, device = _cached_classifier(str(classifier_path.resolve()))
            classifications = classify_pins_batch(cnn, crops, device=device)

    flagged = set(
        cross_validate(rectified, sorted_dets, classifications) if use_ocr else []
    )

    col_indices = {d.column for d in sorted_dets}
    result = SheetResult(
        columns=len(col_indices),
        rows_per_column=max(
            (sum(1 for d in sorted_dets if d.column == c) for c in col_indices),
            default=0,
        ),
    )

    entries = [
        (det.column, det.row, pins)
        for det, (pins, _) in zip(sorted_dets, classifications)
    ]
    scores = (
        boxed_throw_scores(entries)
        if boxed_layout
        else [sum(pins) for _, _, pins in entries]
    )
    for i, (det, (pins, cls_conf), score) in enumerate(
        zip(sorted_dets, classifications, scores)
    ):
        result.throws.append(
            ThrowResult(
                column=det.column,
                row=det.row,
                score=score,
                pins_down=pins,
                confidence=det.confidence,
                classification_confidence=cls_conf,
                ocr_mismatch=i in flagged,
            )
        )
    return result
