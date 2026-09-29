"""Perspective correction and contrast normalization."""

from __future__ import annotations

import cv2
import numpy as np

RECTIFIED_HEIGHT = 1600


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _order_corners(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def find_sheet_quad(gray: np.ndarray) -> np.ndarray | None:
    """Find the four outer sheet corners."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        approximation = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(approximation) == 4 and cv2.contourArea(approximation) >= 0.15 * image_area:
            return _order_corners(approximation.reshape(4, 2).astype(np.float32))
    return None


def rectify_sheet(image: np.ndarray, height: int = RECTIFIED_HEIGHT) -> np.ndarray:
    """Return a perspective-corrected, normalized grayscale sheet."""
    gray = _gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    quad = find_sheet_quad(gray)
    if quad is None:
        return clahe.apply(gray)

    top_left, top_right, bottom_right, bottom_left = quad
    average_width = (
        float(np.linalg.norm(top_right - top_left))
        + float(np.linalg.norm(bottom_right - bottom_left))
    ) / 2
    average_height = (
        float(np.linalg.norm(bottom_left - top_left))
        + float(np.linalg.norm(bottom_right - top_right))
    ) / 2
    width = max(1, round(height * average_width / max(average_height, 1)))
    destination = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    warped = cv2.warpPerspective(
        gray,
        cv2.getPerspectiveTransform(quad, destination),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return clahe.apply(warped)
