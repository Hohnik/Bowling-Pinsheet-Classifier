"""Run both ONNX models and print the detected throw scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pin import DEFAULT_CONFIDENCE, Scanner, draw_detections
from pin.detection import letterbox
from pin.types import Classification, Detection, SheetResult


def contact_sheet(
    crops: list[np.ndarray],
    detections: list[Detection],
    classifications: list[Classification] | None = None,
) -> np.ndarray:
    tile_width = 120
    tile_height = 145 if classifications is not None else 120
    rows = max((detection.row for detection in detections), default=-1) + 1
    columns = max((detection.column for detection in detections), default=-1) + 1
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, np.uint8)
    for index, (crop, detection) in enumerate(zip(crops, detections, strict=True)):
        tile = cv2.resize(crop, (116, 116), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR) if tile.ndim == 2 else tile
        x = detection.column * tile_width + 2
        y = detection.row * tile_height + 2
        canvas[y : y + 116, x : x + 116] = tile
        if classifications is not None:
            pins = "".join(str(pin) for pin in classifications[index].pins)
            cv2.putText(
                canvas,
                pins,
                (x + 4, y + 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 120, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def save_verbose(
    output: Path,
    image: np.ndarray,
    detections: list[Detection],
    crops: list[np.ndarray],
    classifications: list[Classification],
) -> None:
    model_input, _, _, _ = letterbox(image)
    save_image(output / "01-input.jpg", image)
    save_image(output / "02-sheet-model-input.jpg", model_input)
    save_image(output / "03-detected-fields.jpg", draw_detections(image, detections))
    save_image(output / "04-field-crops.jpg", contact_sheet(crops, detections))
    save_image(
        output / "05-classified-pins.jpg",
        contact_sheet(crops, detections, classifications),
    )


def format_throws(result: SheetResult) -> str:
    return " ".join(
        "?" if throw.score is None else str(throw.score) for throw in result.throws
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Score-sheet images")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Save pipeline images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/pipeline"),
        help="Verbose image directory",
    )
    parser.add_argument("--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()

    scanner = Scanner()
    failed = False
    for image_path in args.images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        _, detections = scanner.detect_image(image, args.confidence)
        result, crops, classifications = scanner.classify_detections(image, detections)
        if len(args.images) > 1:
            print(f"{image_path}: {format_throws(result)}")
        else:
            print(format_throws(result))
        failed |= result.needs_review
        if args.verbose:
            destination = args.output / image_path.stem
            save_verbose(destination, image, detections, crops, classifications)
            print(f"Saved pipeline images to {destination}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
