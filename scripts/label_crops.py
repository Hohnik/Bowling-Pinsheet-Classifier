"""Interactively label the nine circles in detected pin-field crops."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pin import Scanner, crop_detections
from pin.classification import classify_circles

LABEL_FIELDS = ("crop", "source", "column", "row", "pins")
PIN_POSITIONS = (
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
WINDOW = "Pin crop labeler"
VIEW_SIZE = 700
HEADER_HEIGHT = 82


@dataclass
class Item:
    crop_path: Path
    source: str
    column: int
    row: int
    image: np.ndarray
    pins: list[int | None]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source, self.column, self.row


def _source_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _crop_name(source: str, column: int, row: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", source).strip("_")
    return f"{slug}_c{column:02d}_r{row:02d}.png"


def _read_labels(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        return {
            (row["source"], int(row["column"]), int(row["row"])): row
            for row in csv.DictReader(file)
        }


def _write_labels(
    path: Path, labels: dict[tuple[str, int, int], dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(
                labels.values(),
                key=lambda row: (
                    row["source"],
                    int(row["column"]),
                    int(row["row"]),
                ),
            )
        )
    temporary.replace(path)


def _nearest_values(
    marks: list[tuple[float, float, float]], maximum_distance: float = 0.20
) -> list[float] | None:
    if len(marks) < 9:
        return None
    values: list[float] = []
    used: set[int] = set()
    for expected_x, expected_y in PIN_POSITIONS:
        choices = [
            ((x - expected_x) ** 2 + (y - expected_y) ** 2, index)
            for index, (x, y, _) in enumerate(marks)
            if index not in used
        ]
        distance, nearest = min(choices)
        if distance > maximum_distance**2:
            return None
        used.add(nearest)
        values.append(marks[nearest][2])
    return values


def _hough_proposal(crop: np.ndarray) -> list[int | None]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape
    size = min(height, width)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=0.10 * size,
        param1=100,
        param2=max(7, 0.055 * size),
        minRadius=round(0.04 * size),
        maxRadius=round(0.11 * size),
    )
    if circles is None:
        return [None] * 9

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    y_coordinates, x_coordinates = np.ogrid[:height, :width]
    sample_radius = max(3, round(0.075 * size))
    marks = []
    for x, y, _ in circles[0]:
        sample = (x_coordinates - x) ** 2 + (y_coordinates - y) ** 2
        density = float(np.mean(binary[sample <= sample_radius**2] > 0))
        marks.append((x / width, y / height, density))
    values = _nearest_values(marks)
    return [int(value >= 0.78) for value in values] if values else [None] * 9


def _proposed_pins(crop: np.ndarray) -> list[int | None]:
    classification = classify_circles(crop)
    if classification is not None and not classification.needs_review:
        return list(classification.pins)
    return _hough_proposal(crop)


def _make_item(
    source_path: Path,
    column: int,
    row: int,
    crop: np.ndarray,
    crop_directory: Path,
    existing: dict[str, str] | None,
) -> Item:
    source = _source_name(source_path)
    crop_path = crop_directory / _crop_name(source, column, row)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Could not write {crop_path}")

    pins = (
        _proposed_pins(crop)
        if existing is None
        else [int(value) for value in existing["pins"]]
    )
    return Item(crop_path, source, column, row, crop, pins)


def _render(
    item: Item, index: int, count: int, message: str
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((HEADER_HEIGHT + VIEW_SIZE, VIEW_SIZE, 3), 245, dtype=np.uint8)
    title = f"{index + 1}/{count}  {item.source}  C{item.column:02d} R{item.row:02d}"
    cv2.putText(
        canvas,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "1-9/click | 0 empty | A filled | U unknown | Space save | Backspace back | Q quit",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    bits = "".join("?" if value is None else str(value) for value in item.pins)
    status = f"pins: {bits}"
    cv2.putText(
        canvas,
        status,
        (12, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 100, 20),
        1,
        cv2.LINE_AA,
    )
    if message:
        cv2.putText(
            canvas,
            message,
            (430, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 220),
            1,
            cv2.LINE_AA,
        )

    height, width = item.image.shape[:2]
    scale = min(VIEW_SIZE / width, VIEW_SIZE / height)
    display_width, display_height = round(width * scale), round(height * scale)
    left = (VIEW_SIZE - display_width) // 2
    top = HEADER_HEIGHT + (VIEW_SIZE - display_height) // 2
    resized = cv2.resize(
        item.image, (display_width, display_height), interpolation=cv2.INTER_NEAREST
    )
    canvas[top : top + display_height, left : left + display_width] = resized

    radius = max(15, round(min(display_width, display_height) * 0.055))
    for number, ((x, y), value) in enumerate(zip(PIN_POSITIONS, item.pins), 1):
        center = (left + round(x * display_width), top + round(y * display_height))
        color = (
            (0, 190, 0)
            if value == 1
            else (220, 120, 0)
            if value == 0
            else (0, 120, 255)
        )
        cv2.circle(canvas, center, radius, color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(number),
            (center[0] - 6, center[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas, (left, top, display_width, display_height)


def _save_item(
    item: Item,
    labels: dict[tuple[str, int, int], dict[str, str]],
    output: Path,
) -> None:
    labels[item.key] = {
        "crop": item.crop_path.as_posix(),
        "source": item.source,
        "column": str(item.column),
        "row": str(item.row),
        "pins": "".join(str(value) for value in item.pins),
    }
    _write_labels(output, labels)


def _handle_key(key: int, item: Item) -> str:
    if ord("1") <= key <= ord("9"):
        pin = key - ord("1")
        value = item.pins[pin]
        item.pins[pin] = 1 if value is None else 1 - value
    elif key == ord("0"):
        item.pins = [0] * 9
    elif key in (ord("a"), ord("A")):
        item.pins = [1] * 9
    elif key in (ord("u"), ord("U")):
        item.pins = [None] * 9
    elif key in (10, 13, 32):
        return "save"
    elif key in (8, 127):
        return "back"
    elif key in (ord("q"), ord("Q"), 27):
        return "quit"
    return ""


def label(  # noqa: C901
    items: list[Item], labels: dict[tuple[str, int, int], dict[str, str]], output: Path
) -> None:
    if not items:
        print("No unlabeled crops. Use --review to inspect existing labels.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, VIEW_SIZE, HEADER_HEIGHT + VIEW_SIZE)
    index = 0
    message = ""
    bounds = (0, 0, 1, 1)

    def click(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal bounds, message
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        left, top, width, height = bounds
        if not (left <= x <= left + width and top <= y <= top + height):
            return
        normalized = ((x - left) / width, (y - top) / height)
        distance, pin = min(
            ((normalized[0] - px) ** 2 + (normalized[1] - py) ** 2, pin)
            for pin, (px, py) in enumerate(PIN_POSITIONS)
        )
        if distance <= 0.10**2:
            value = items[index].pins[pin]
            items[index].pins[pin] = 1 if value is None else 1 - value
            message = ""
            updated, bounds = _render(items[index], index, len(items), message)
            cv2.imshow(WINDOW, updated)

    cv2.setMouseCallback(WINDOW, click)
    while 0 <= index < len(items):
        canvas, bounds = _render(items[index], index, len(items), message)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(0) & 0xFF
        message = ""
        action = _handle_key(key, items[index])
        if action == "save":
            if any(value is None for value in items[index].pins):
                message = "Label all nine pins"
                continue
            _save_item(items[index], labels, output)
            index += 1
        elif action == "back":
            index = max(0, index - 1)
        elif action == "quit":
            break

    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("images", nargs="+", type=Path, help="Score-sheet images")
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/pin_classifier/labels.csv"),
    )
    _ = parser.add_argument("--crop-directory", type=Path, default=Path("data/crops"))
    _ = parser.add_argument(
        "--review", action="store_true", help="Include already labeled crops"
    )
    args = parser.parse_args()

    labels = _read_labels(args.output)
    scanner = Scanner()
    items: list[Item] = []
    for image_path in args.images:
        sheet, detections = scanner.detect_sheet(image_path)
        for detection, crop in zip(detections, crop_detections(sheet, detections)):
            key = (_source_name(image_path), detection.column, detection.row)
            existing = labels.get(key)
            if existing is not None and not args.review:
                continue
            items.append(
                _make_item(
                    image_path,
                    detection.column,
                    detection.row,
                    crop,
                    args.crop_directory,
                    existing,
                )
            )

    label(items, labels, args.output)


if __name__ == "__main__":
    main()
