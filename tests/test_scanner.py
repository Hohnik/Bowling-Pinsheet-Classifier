from pathlib import Path

import pytest

from pin import Scanner
from pin.classification import CIRCLE_MARGIN, CIRCLE_THRESHOLD, states

SHEETS = Path("data/sheets")


@pytest.fixture(scope="module")
def scanner() -> Scanner:
    return Scanner()


@pytest.mark.parametrize(
    ("filename", "diagrams", "total"),
    [
        ("old/001.jpeg", 90, 335),
        ("old/002.jpeg", 120, 472),
        ("old/003.jpeg", 120, 454),
        ("old/004.jpeg", 30, 108),
        ("old/005.jpeg", 120, 452),
        ("old/006.jpeg", 120, 623),
        ("old/007.jpeg", 120, 499),
        ("new/001.jpeg", 120, 604),
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
    "filename",
    ["003.png", "004.png", "005.png", "007.png", "012.png"],
)
def test_boxed_grid_is_completed(scanner: Scanner, filename: str) -> None:
    _, detections = scanner.detect_sheet(SHEETS / "new" / filename)
    assert len(detections) == 120
    assert {(item.column, item.row) for item in detections} == {
        (column, row) for column in range(15) for row in range(8)
    }


def test_uncertain_values_are_rejected() -> None:
    pins, confidence = states(
        [CIRCLE_THRESHOLD + CIRCLE_MARGIN / 2], CIRCLE_THRESHOLD, CIRCLE_MARGIN
    )
    assert pins == (None,)
    assert confidence < 1
