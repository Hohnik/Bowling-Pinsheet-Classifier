"""Geometry-based classification for nine-pin diagram symbols."""

from __future__ import annotations

import cv2
import numpy as np

_PIN_POSITIONS = [
    (0.50, 0.80),
    (0.31, 0.63),
    (0.69, 0.63),
    (0.13, 0.46),
    (0.50, 0.46),
    (0.87, 0.46),
    (0.31, 0.29),
    (0.69, 0.29),
    (0.50, 0.12),
]


def classify_circle_pins(crop: np.ndarray) -> tuple[list[int], float] | None:
    """Classify a boxed diagram, or return ``None`` for another layout."""
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
    if horizontal < 0.4 or vertical < 0.4:
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
            area = stats[index, cv2.CC_STAT_AREA]
            marks.append(
                (
                    centroids[index, 0] / width,
                    centroids[index, 1] / height,
                    area / (mark_width * mark_height),
                )
            )
    if len(marks) < 9:
        return None

    pins: list[int] = []
    densities: list[float] = []
    used: set[int] = set()
    for expected_x, expected_y in _PIN_POSITIONS:
        choices = [
            ((x - expected_x) ** 2 + (y - expected_y) ** 2, index)
            for index, (x, y, _) in enumerate(marks)
            if index not in used
        ]
        distance_sq, nearest = min(choices)
        if distance_sq > 0.13**2:
            return None
        used.add(nearest)
        density = marks[nearest][2]
        densities.append(density)
        pins.append(int(density >= 0.67))

    confidence = min(1.0, float(np.mean(np.abs(np.array(densities) - 0.67))) * 4)
    return pins, confidence


def classify_dash_pins(crop: np.ndarray) -> tuple[list[int], float] | None:
    """Classify the dot-and-dash symbols used by older sheets."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    marks: list[tuple[float, float, float]] = []
    for index in range(1, count):
        mark_width = stats[index, cv2.CC_STAT_WIDTH]
        mark_height = stats[index, cv2.CC_STAT_HEIGHT]
        area = stats[index, cv2.CC_STAT_AREA]
        if (
            0.04 * width <= mark_width <= 0.35 * width
            and 0.03 * height <= mark_height <= 0.30 * height
            and area >= 3
            and mark_width / mark_height <= 6
        ):
            marks.append(
                (
                    centroids[index, 0] / width,
                    centroids[index, 1] / height,
                    mark_width / mark_height,
                )
            )
    if len(marks) < 9:
        return None

    pins: list[int] = []
    ratios: list[float] = []
    used: set[int] = set()
    for expected_x, expected_y in _PIN_POSITIONS:
        choices = [
            ((x - expected_x) ** 2 + (y - expected_y) ** 2, index)
            for index, (x, y, _) in enumerate(marks)
            if index not in used
        ]
        distance_sq, nearest = min(choices)
        if distance_sq > 0.15**2:
            return None
        used.add(nearest)
        ratio = marks[nearest][2]
        ratios.append(ratio)
        pins.append(int(ratio >= 1.75))

    confidence = min(1.0, min(abs(ratio - 1.75) for ratio in ratios) / 0.75)
    return pins, confidence


def classify_pin_symbols(crop: np.ndarray) -> tuple[list[int], float] | None:
    """Classify either supported symbol style."""
    return classify_circle_pins(crop) or classify_dash_pins(crop)


def boxed_throw_scores(entries: list[tuple[int, int, list[int]]]) -> list[int]:
    """Convert cumulative clearing diagrams into per-throw scores.

    Boxed sheets alternate a direct-count ``Volle`` row and a cumulative
    ``Abräumen`` row. A lower count, or the throw after all nine pins, starts a
    new figure.
    """
    scores = [sum(pins) for _, _, pins in entries]
    row_numbers = {row for _, row, _ in entries}
    for row in row_numbers:
        if row % 2 == 0:
            continue
        previous = 0
        indices = sorted(
            (
                index
                for index, (_, entry_row, _) in enumerate(entries)
                if entry_row == row
            ),
            key=lambda index: entries[index][0],
        )
        for index in indices:
            count = sum(entries[index][2])
            scores[index] = (
                count if previous == 9 or count < previous else count - previous
            )
            previous = count
    return scores
