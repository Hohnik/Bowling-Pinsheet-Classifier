from pathlib import Path

import numpy as np
import pytest

from pin import DEFAULT_CLASSIFIER_PATH, DEFAULT_EXTRACTOR_PATH, Detection, Scanner
from pin.classification import CIRCLE_MARGIN, CIRCLE_THRESHOLD, states
from pin.detection import letterbox, sort_detections

SHEETS = Path("data/sheets")


@pytest.fixture(scope="module")
def scanner() -> Scanner:
    if not DEFAULT_EXTRACTOR_PATH.exists() or not DEFAULT_CLASSIFIER_PATH.exists():
        pytest.skip("Train and export both ONNX models first")
    return Scanner()


@pytest.mark.parametrize(
    ("filename", "diagrams", "total"),
    [
        ("000.png", 120, 604),
        ("001.jpeg", 120, 604),
        ("007.png", 120, 534),
    ],
)
def test_sample_sheet(
    scanner: Scanner, filename: str, diagrams: int, total: int
) -> None:
    result = scanner.scan_sheet(SHEETS / filename)
    assert len(result.throws) == diagrams
    assert result.total_pins == total
    assert not result.needs_review


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("003.png", 120),
        ("004.png", 104),
        ("005.png", 120),
        ("007.png", 120),
        ("012.png", 120),
    ],
)
def test_boxed_grid_is_detected(scanner: Scanner, filename: str, expected: int) -> None:
    _, detections = scanner.detect_sheet(SHEETS / filename)
    assert len(detections) == expected
    if expected == 120:
        assert {(item.column, item.row) for item in detections} == {
            (column, row) for column in range(15) for row in range(8)
        }


def test_letterbox_preserves_aspect_ratio() -> None:
    image = np.zeros((600, 1200, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image)
    assert canvas.shape == (1280, 1280, 3)
    assert scale == pytest.approx(1280 / 1200)
    assert pad_x == 0
    assert pad_y == 320


def test_detections_are_sorted_into_the_grid() -> None:
    detections = [
        Detection(
            x_center=100 + column * 20,
            y_center=100 + row * 30 + column * 2,
            width=18,
            height=20,
            confidence=0.9,
        )
        for row in range(8)
        for column in range(15)
    ]
    ordered = sort_detections(list(reversed(detections)))
    assert [(item.column, item.row) for item in ordered] == [
        (column, row) for column in range(15) for row in range(8)
    ]


def test_uncertain_values_are_rejected() -> None:
    pins, confidence = states(
        [CIRCLE_THRESHOLD + CIRCLE_MARGIN / 2], CIRCLE_THRESHOLD, CIRCLE_MARGIN
    )
    assert pins == (None,)
    assert confidence < 1
