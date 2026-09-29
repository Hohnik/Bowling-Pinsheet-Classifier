"""Interactively label pin-field boxes on score-sheet images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pin import Scanner

WINDOW = "Sheet field labeler"
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 900
HEADER_HEIGHT = 58


@dataclass
class SheetItem:
    image_path: Path
    annotation_path: Path
    image: np.ndarray
    boxes: list[tuple[float, float, float, float]]


def image_key(path: Path) -> str:
    return f"{path.stem}_{path.suffix[1:].lower()}"


def read_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        _, x, y, width, height = line.split()
        boxes.append((float(x), float(y), float(width), float(height)))
    return boxes


def write_boxes(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n"
            for x, y, width, height in sorted(boxes, key=lambda box: (box[0], box[1]))
        )
    )


def proposed_boxes(
    scanner: Scanner, image: np.ndarray
) -> list[tuple[float, float, float, float]]:
    _, detections = scanner.detect_image(image, confidence=0.50)
    height, width = image.shape[:2]
    return [
        (
            detection.x_center / width,
            detection.y_center / height,
            detection.width / width,
            detection.height / height,
        )
        for detection in detections
    ]


def render(
    item: SheetItem,
    index: int,
    count: int,
    start: tuple[int, int] | None = None,
    cursor: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((HEADER_HEIGHT + CANVAS_HEIGHT, CANVAS_WIDTH, 3), 245, np.uint8)
    cv2.putText(
        canvas,
        f"{index + 1}/{count}  {item.image_path}  boxes={len(item.boxes)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Drag: add box | Right-click: delete box | U: clear | Space: save | Backspace: back | Q: quit",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    height, width = item.image.shape[:2]
    scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height)
    display_width, display_height = round(width * scale), round(height * scale)
    left = (CANVAS_WIDTH - display_width) // 2
    top = HEADER_HEIGHT + (CANVAS_HEIGHT - display_height) // 2
    resized = cv2.resize(
        item.image, (display_width, display_height), interpolation=cv2.INTER_AREA
    )
    canvas[top : top + display_height, left : left + display_width] = resized
    for x, y, box_width, box_height in item.boxes:
        x1 = left + round((x - box_width / 2) * display_width)
        y1 = top + round((y - box_height / 2) * display_height)
        x2 = left + round((x + box_width / 2) * display_width)
        y2 = top + round((y + box_height / 2) * display_height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
    if start is not None and cursor is not None:
        cv2.rectangle(canvas, start, cursor, (0, 120, 255), 2)
    return canvas, (left, top, display_width, display_height)


def label(items: list[SheetItem]) -> None:  # noqa: C901
    if not items:
        print("No unlabeled sheets. Use --review to inspect existing annotations.")
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_WIDTH, HEADER_HEIGHT + CANVAS_HEIGHT)
    index = 0
    bounds = (0, 0, 1, 1)
    drag_start: tuple[int, int] | None = None
    cursor: tuple[int, int] | None = None

    def redraw() -> None:
        nonlocal bounds
        canvas, bounds = render(items[index], index, len(items), drag_start, cursor)
        cv2.imshow(WINDOW, canvas)

    def mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal cursor, drag_start
        left, top, width, height = bounds
        inside = left <= x <= left + width and top <= y <= top + height
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            drag_start = (x, y)
            cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            cursor = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1, x2 = sorted((drag_start[0], min(max(x, left), left + width)))
            y1, y2 = sorted((drag_start[1], min(max(y, top), top + height)))
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                items[index].boxes.append(
                    (
                        ((x1 + x2) / 2 - left) / width,
                        ((y1 + y2) / 2 - top) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
            drag_start = cursor = None
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and inside and items[index].boxes:
            normalized_x, normalized_y = (x - left) / width, (y - top) / height
            candidates = [
                (box_width * box_height, box_index)
                for box_index, (center_x, center_y, box_width, box_height) in enumerate(
                    items[index].boxes
                )
                if abs(normalized_x - center_x) <= box_width / 2
                and abs(normalized_y - center_y) <= box_height / 2
            ]
            if candidates:
                items[index].boxes.pop(min(candidates)[1])
                redraw()

    cv2.setMouseCallback(WINDOW, mouse)
    while 0 <= index < len(items):
        redraw()
        key = cv2.waitKey(0) & 0xFF
        if key in (10, 13, 32):
            write_boxes(items[index].annotation_path, items[index].boxes)
            index += 1
        elif key in (8, 127):
            index = max(0, index - 1)
        elif key in (ord("u"), ord("U")):
            items[index].boxes.clear()
        elif key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/field_extractor"),
    )
    parser.add_argument(
        "--review", action="store_true", help="Include already labeled sheets"
    )
    args = parser.parse_args()

    scanner = Scanner()
    items = []
    for image_path in args.images:
        annotation = args.annotations / f"{image_key(image_path)}.txt"
        if annotation.exists() and not args.review:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        boxes = (
            read_boxes(annotation)
            if annotation.exists()
            else proposed_boxes(scanner, image)
        )
        items.append(SheetItem(image_path, annotation, image, boxes))
    label(items)


if __name__ == "__main__":
    main()
