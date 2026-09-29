"""Draw every detected pin-diagram box on a sheet image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2

from pin import Scanner, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("image", type=Path, help="Score-sheet image")
    _ = parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("detections.png"),
        help="Annotated image path",
    )
    _ = parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum diagram detection confidence",
    )
    args = parser.parse_args()
    image = cast("Path", args.image)
    output = cast("Path", args.output)
    confidence = cast("float", args.confidence)

    sheet, detections = Scanner().detect_sheet(image, confidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw_detections(sheet, detections)):
        raise RuntimeError(f"Could not write {output}")
    print(f"Detected {len(detections)} diagrams -> {output}")


if __name__ == "__main__":
    main()
