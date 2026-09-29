"""YOLO pin-field detection through OpenCV DNN."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from pin.types import Detection

YOLO_INPUT_SIZE = 1280
EXPECTED_COLUMNS = 15
EXPECTED_ROWS = 8
EXPECTED_FIELDS = EXPECTED_COLUMNS * EXPECTED_ROWS
DEFAULT_CONFIDENCE = 0.85


def letterbox(
    image: np.ndarray, size: int = YOLO_INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """Resize and pad an image to the square YOLO input shape."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


class YoloDetector:
    """Run an exported single-class YOLO model with OpenCV."""

    def __init__(
        self, model_path: Path | str, input_size: int = YOLO_INPUT_SIZE
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Field extractor not found: {path}. Run `just train-sheets`."
            )
        self.network = cv2.dnn.readNetFromONNX(str(path))
        self.input_size = input_size

    def detect(
        self,
        image: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = 0.5,
    ) -> list[Detection]:
        """Detect pin fields and map their boxes to source-image coordinates."""
        height, width = image.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1 / 255,
            size=(self.input_size, self.input_size),
            swapRB=True,
        )
        self.network.setInput(blob)
        predictions = np.squeeze(self.network.forward())
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[float]] = []
        scores: list[float] = []
        for prediction in predictions:
            center_x, center_y, box_width, box_height = prediction[:4]
            score = float(np.max(prediction[4:]))
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
            scores.append(score)

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


def _deskewed_centers(
    detections: list[Detection],
) -> list[tuple[Detection, float, float]]:
    median_width = float(np.median([item.width for item in detections]))
    median_height = float(np.median([item.height for item in detections]))
    slopes = [
        (second.y_center - first.y_center) / (second.x_center - first.x_center)
        for first, second in combinations(detections, 2)
        if 0.6 * median_width
        < abs(second.x_center - first.x_center)
        < 1.5 * median_width
        and abs(second.y_center - first.y_center) < 0.8 * median_height
    ]
    angle = np.arctan(float(np.median(slopes))) if slopes else 0.0
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return [
        (
            item,
            cosine * item.x_center + sine * item.y_center,
            -sine * item.x_center + cosine * item.y_center,
        )
        for item in detections
    ]


def sort_detections(detections: list[Detection]) -> list[Detection]:
    """Assign the model boxes to the sheet's 15 by 8 reading order."""
    if not detections:
        return []

    centers = _deskewed_centers(detections)
    if len(centers) == EXPECTED_FIELDS:
        ordered = sorted(centers, key=lambda item: item[2])
        rows = [
            ordered[start : start + EXPECTED_COLUMNS]
            for start in range(0, EXPECTED_FIELDS, EXPECTED_COLUMNS)
        ]
    else:
        median_height = float(np.median([item.height for item in detections]))
        ordered = sorted(centers, key=lambda item: item[2])
        rows: list[list[tuple[Detection, float, float]]] = [[ordered[0]]]
        for item in ordered[1:]:
            if item[2] - rows[-1][-1][2] > median_height * 0.5:
                rows.append([item])
            else:
                rows[-1].append(item)

    result: list[Detection] = []
    for row_index, row in enumerate(rows):
        for column_index, (detection, _, _) in enumerate(
            sorted(row, key=lambda item: item[1])
        ):
            detection.column = column_index
            detection.row = row_index
            result.append(detection)
    return sorted(result, key=lambda item: (item.column, item.row))


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
    thickness = max(2, round(image.shape[1] / 1200))
    for detection in detections:
        cv2.rectangle(
            output,
            (detection.x_min, detection.y_min),
            (detection.x_max, detection.y_max),
            (0, 255, 0),
            thickness,
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
