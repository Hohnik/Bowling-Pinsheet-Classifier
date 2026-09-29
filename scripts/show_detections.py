"""Draw every pin-diagram box found by the production pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detect import draw_detections  # noqa: E402
from pipeline import detect_sheet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Score-sheet image")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG or JPEG")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    output = args.output or Path(f"{args.image.stem}_detections.png")
    sheet, detections = detect_sheet(args.image, confidence=args.confidence)
    if sheet.ndim == 2:
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw_detections(sheet, detections)):
        raise RuntimeError(f"Could not write {output}")
    print(f"Detected {len(detections)} diagrams -> {output}")


if __name__ == "__main__":
    main()
