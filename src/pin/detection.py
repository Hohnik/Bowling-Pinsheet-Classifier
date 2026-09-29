"""Pin-diagram detection with boxed-cell geometry and ONNX YOLO."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from pin.types import Detection

YOLO_INPUT_SIZE = 1280
BOXED_COLUMNS = 15
BOXED_ROWS = 8


class OnnxDetector:
    """Small YOLO detector executed by OpenCV DNN."""

    def __init__(
        self, model_path: Path | str, input_size: int = YOLO_INPUT_SIZE
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Detector model not found: {path}")
        self.network = cv2.dnn.readNetFromONNX(str(path))
        self.input_size = input_size

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> list[Detection]:
        """Run YOLO and return detections in source-image coordinates."""
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1 / 255, (self.input_size, self.input_size), swapRB=True
        )
        self.network.setInput(blob)
        output = self.network.forward()[0]
        predictions = output.T if output.shape[0] == 5 else output

        boxes: list[list[float]] = []
        scores: list[float] = []
        for center_x, center_y, box_width, box_height, score in predictions:
            if score < confidence:
                continue
            boxes.append(
                [
                    (float(center_x - box_width / 2) - pad_x) / scale,
                    (float(center_y - box_height / 2) - pad_y) / scale,
                    float(box_width) / scale,
                    float(box_height) / scale,
                ]
            )
            scores.append(float(score))

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
        detections: list[Detection] = []
        for index in np.asarray(indices).reshape(-1):
            x, y, box_width, box_height = boxes[int(index)]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1 = min(float(width), x + box_width)
            y1 = min(float(height), y + box_height)
            detections.append(
                Detection(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    x1 - x0,
                    y1 - y0,
                    scores[int(index)],
                )
            )
        return detections


def _cluster_axis(
    items: list[tuple[Detection, float, float]],
    coordinate: int,
    threshold: float,
) -> list[list[tuple[Detection, float, float]]]:
    ordered = sorted(items, key=lambda item: item[coordinate])
    groups = [[ordered[0]]]
    for item in ordered[1:]:
        if item[coordinate] - groups[-1][-1][coordinate] > threshold:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _remove_extra_columns(
    groups: list[list[tuple[Detection, float, float]]],
) -> list[list[tuple[Detection, float, float]]]:
    """Remove isolated boxes visible from sheets underneath the target sheet."""
    while len(groups) > BOXED_COLUMNS:
        centers = [float(np.median([item[1] for item in group])) for group in groups]
        gaps = np.diff(centers)
        largest_index = int(np.argmax(gaps))
        typical_gap = float(np.median(np.delete(gaps, largest_index)))
        excess = len(groups) - BOXED_COLUMNS
        left_count = largest_index + 1
        right_count = len(groups) - left_count
        if gaps[largest_index] < 1.5 * typical_gap:
            break
        if left_count <= excess:
            groups = groups[left_count:]
        elif right_count <= excess:
            groups = groups[:left_count]
        else:
            break
    return groups


def _complete_grid_row(
    row: int,
    assigned: dict[tuple[int, int], Detection],
) -> list[Detection] | None:
    observed = [
        (column, assigned[(column, row)])
        for column in range(BOXED_COLUMNS)
        if (column, row) in assigned
    ]
    if len(observed) < BOXED_COLUMNS // 2:
        return None

    observed_columns = np.asarray([column for column, _ in observed])
    x_fit = np.polyfit(
        observed_columns,
        [item.x_center for _, item in observed],
        1,
    )
    y_fit = np.polyfit(
        observed_columns,
        [item.y_center for _, item in observed],
        1,
    )
    row_width = float(np.median([item.width for _, item in observed]))
    row_height = float(np.median([item.height for _, item in observed]))
    fitted_x = np.polyval(x_fit, np.arange(BOXED_COLUMNS))
    fitted_y = np.polyval(y_fit, np.arange(BOXED_COLUMNS))

    completed: list[Detection] = []
    for column in range(BOXED_COLUMNS):
        detection = assigned.get(
            (column, row),
            Detection(
                float(fitted_x[column]),
                float(fitted_y[column]),
                row_width,
                row_height,
                0.0,
            ),
        )
        detection.column = column
        detection.row = row
        completed.append(detection)
    return completed


def _complete_bordered_grid(candidates: list[Detection]) -> list[Detection]:
    """Complete the fixed 15 by 8 grid when some cell borders are obscured."""
    if len(candidates) < BOXED_COLUMNS * 4:
        return candidates

    median_width = float(np.median([item.width for item in candidates]))
    median_height = float(np.median([item.height for item in candidates]))
    candidates = [
        item
        for item in candidates
        if 0.7 * median_width <= item.width <= 1.3 * median_width
        and 0.7 * median_height <= item.height <= 1.3 * median_height
    ]

    slopes = [
        (second.y_center - first.y_center) / (second.x_center - first.x_center)
        for first, second in combinations(candidates, 2)
        if 0.6 * median_width
        < abs(second.x_center - first.x_center)
        < 1.5 * median_width
        and abs(second.y_center - first.y_center) < 0.8 * median_height
    ]
    if not slopes:
        return candidates

    angle = np.arctan(float(np.median(slopes)))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotated = [
        (
            item,
            cosine * item.x_center + sine * item.y_center,
            -sine * item.x_center + cosine * item.y_center,
        )
        for item in candidates
    ]
    columns = _remove_extra_columns(_cluster_axis(rotated, 1, median_width * 0.5))
    rows = _cluster_axis(rotated, 2, median_height * 0.5)
    if len(columns) != BOXED_COLUMNS or len(rows) != BOXED_ROWS:
        return candidates

    column_centers = [
        float(np.median([item[1] for item in group])) for group in columns
    ]
    row_centers = [float(np.median([item[2] for item in group])) for group in rows]
    column_items = [{id(item[0]) for item in group} for group in columns]
    row_items = [{id(item[0]) for item in group} for group in rows]
    assigned: dict[tuple[int, int], Detection] = {}
    for detection, rotated_x, rotated_y in rotated:
        column = int(np.argmin(np.abs(np.asarray(column_centers) - rotated_x)))
        row = int(np.argmin(np.abs(np.asarray(row_centers) - rotated_y)))
        if (
            id(detection) not in column_items[column]
            or id(detection) not in row_items[row]
        ):
            continue
        assigned[(column, row)] = detection

    completed: list[Detection] = []
    for row in range(BOXED_ROWS):
        completed_row = _complete_grid_row(row, assigned)
        if completed_row is None:
            return candidates
        completed.extend(completed_row)
    return completed


def detect_bordered_diagrams(image: np.ndarray) -> list[Detection]:
    """Detect diagrams enclosed by individual rectangular cells."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, width = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_size = max(3, round(width * 0.002))
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Detection] = []
    minimum_side, maximum_side = width * 0.015, width * 0.12
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (
            minimum_side <= box_width <= maximum_side
            and minimum_side <= box_height <= maximum_side
            and 0.5 <= box_width / box_height <= 1.8
            and cv2.contourArea(contour) >= 0.35 * box_width * box_height
        ):
            continue

        crop = binary[y : y + box_height, x : x + box_width]
        count, _, stats, _ = cv2.connectedComponentsWithStats(crop)
        round_marks = sum(
            0.05 * box_width <= stats[index, cv2.CC_STAT_WIDTH] <= 0.3 * box_width
            and 0.05 * box_height
            <= stats[index, cv2.CC_STAT_HEIGHT]
            <= 0.3 * box_height
            and 0.35
            <= stats[index, cv2.CC_STAT_WIDTH] / stats[index, cv2.CC_STAT_HEIGHT]
            <= 2.0
            for index in range(1, count)
        )
        if round_marks >= 7:
            candidates.append(
                Detection(
                    x + box_width / 2,
                    y + box_height / 2,
                    float(box_width),
                    float(box_height),
                    1.0,
                )
            )

    kept: list[Detection] = []
    for detection in sorted(
        candidates, key=lambda item: item.width * item.height, reverse=True
    ):
        overlaps = any(
            abs(detection.x_center - other.x_center)
            < min(detection.width, other.width) / 2
            and abs(detection.y_center - other.y_center)
            < min(detection.height, other.height) / 2
            for other in kept
        )
        if not overlaps:
            kept.append(detection)
    return _complete_bordered_grid(kept)


def sort_detections(detections: list[Detection]) -> list[Detection]:
    """Assign columns and rows, then return column-major reading order."""
    if not detections:
        return []
    if all(item.column >= 0 and item.row >= 0 for item in detections):
        return sorted(detections, key=lambda item: (item.column, item.row))
    by_x = sorted(detections, key=lambda item: item.x_center)
    threshold = float(np.median([item.width for item in by_x])) * 0.5
    columns: list[list[Detection]] = [[by_x[0]]]
    for detection in by_x[1:]:
        if detection.x_center - columns[-1][-1].x_center > threshold:
            columns.append([detection])
        else:
            columns[-1].append(detection)

    ordered: list[Detection] = []
    for column_index, column in enumerate(columns):
        for row_index, detection in enumerate(
            sorted(column, key=lambda item: item.y_center)
        ):
            detection.column = column_index
            detection.row = row_index
            ordered.append(detection)
    return ordered


def crop_detections(
    image: np.ndarray, detections: list[Detection], padding: int = 2
) -> list[np.ndarray]:
    """Crop detected regions from an image."""
    height, width = image.shape[:2]
    return [
        image[
            max(0, detection.y_min - padding) : min(height, detection.y_max + padding),
            max(0, detection.x_min - padding) : min(width, detection.x_max + padding),
        ].copy()
        for detection in detections
    ]


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection boxes on a copy of an image."""
    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        cv2.rectangle(
            output,
            (detection.x_min, detection.y_min),
            (detection.x_max, detection.y_max),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output,
            f"c{detection.column}r{detection.row} {detection.confidence:.2f}",
            (detection.x_min, detection.y_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
    return output
