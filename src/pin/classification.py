"""Deterministic pin-symbol classification with uncertainty rejection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pin.types import Classification, total

CIRCLE_THRESHOLD = 0.646
CIRCLE_MARGIN = 0.02
CROP_INPUT_SIZE = 128
CROP_THRESHOLD = 0.60

_PIN_POSITIONS = (
    (0.50, 0.80),
    (0.31, 0.63),
    (0.69, 0.63),
    (0.13, 0.46),
    (0.50, 0.46),
    (0.87, 0.46),
    (0.31, 0.29),
    (0.69, 0.29),
    (0.50, 0.12),
)


def _assign_positions(
    marks: list[tuple[float, float, float]], maximum_distance: float
) -> list[float] | None:
    if len(marks) < 9:
        return None
    values: list[float] = []
    used: set[int] = set()
    for expected_x, expected_y in _PIN_POSITIONS:
        choices = [
            ((x - expected_x) ** 2 + (y - expected_y) ** 2, index)
            for index, (x, y, _) in enumerate(marks)
            if index not in used
        ]
        distance_squared, nearest = min(choices)
        if distance_squared > maximum_distance**2:
            return None
        used.add(nearest)
        values.append(marks[nearest][2])
    return values


def states(
    values: list[float], threshold: float, margin: float
) -> tuple[tuple[int | None, ...], float]:
    distances = [abs(value - threshold) for value in values]
    pins = tuple(
        None if distance <= margin else int(value >= threshold)
        for value, distance in zip(values, distances)
    )
    confidence = min(1.0, min(distances) / (margin * 3))
    return pins, confidence


def classify_circles(crop: np.ndarray) -> Classification | None:
    """Classify filled and hollow circles in a boxed diagram."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape

    horizontal = max(
        np.mean(binary > 0, axis=1)[: max(1, height // 5)].max(),
        np.mean(binary > 0, axis=1)[-max(1, height // 5) :].max(),
    )
    vertical = max(
        np.mean(binary > 0, axis=0)[: max(1, width // 5)].max(),
        np.mean(binary > 0, axis=0)[-max(1, width // 5) :].max(),
    )
    if horizontal < 0.35 or vertical < 0.35:
        return None

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    marks: list[tuple[float, float, float]] = []
    for index in range(1, count):
        mark_width = stats[index, cv2.CC_STAT_WIDTH]
        mark_height = stats[index, cv2.CC_STAT_HEIGHT]
        if (
            0.07 * width <= mark_width <= 0.25 * width
            and 0.07 * height <= mark_height <= 0.25 * height
            and 0.45 <= mark_width / mark_height <= 1.8
        ):
            marks.append(
                (
                    centroids[index, 0] / width,
                    centroids[index, 1] / height,
                    stats[index, cv2.CC_STAT_AREA] / (mark_width * mark_height),
                )
            )

    densities = _assign_positions(marks, maximum_distance=0.13)
    if densities is None:
        return None
    pins, confidence = states(densities, CIRCLE_THRESHOLD, CIRCLE_MARGIN)
    return Classification(pins, confidence, "circles")


class CropClassifier:
    """Classify the nine pin states with an OpenCV ONNX model."""

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pin classifier not found: {path}. Run `just train-crops`."
            )
        self.network = cv2.dnn.readNetFromONNX(str(path))

    def classify(self, crop: np.ndarray) -> Classification:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blob = cv2.dnn.blobFromImage(
            gray,
            scalefactor=1 / 255,
            size=(CROP_INPUT_SIZE, CROP_INPUT_SIZE),
        )
        self.network.setInput(blob)
        logits = self.network.forward().reshape(-1)
        probabilities = 1 / (1 + np.exp(-logits))
        pins = tuple(int(value >= CROP_THRESHOLD) for value in probabilities)
        confidence = min(1.0, float(np.min(abs(probabilities - CROP_THRESHOLD))) * 4)
        return Classification(pins, confidence, "onnx")


def boxed_throw_scores(
    entries: list[tuple[int, int, tuple[int | None, ...]]],
) -> list[int | None]:
    """Convert cumulative clearing diagrams into per-throw scores."""
    scores = [total(pins) for _, _, pins in entries]
    for row in {row for _, row, _ in entries}:
        if row % 2 == 0:
            continue
        previous: int | None = 0
        indices = sorted(
            (
                index
                for index, (_, entry_row, _) in enumerate(entries)
                if entry_row == row
            ),
            key=lambda index: entries[index][0],
        )
        for index in indices:
            pins = entries[index][2]
            count = total(pins)
            if count is None or previous is None:
                scores[index] = None
            else:
                scores[index] = (
                    count if previous == 9 or count < previous else count - previous
                )
            previous = count
    return scores
