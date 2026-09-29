"""Scan one or more score-sheet images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from pin import Scanner, SheetResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Score-sheet image(s) path e.g. data/sheets/*.jpg",
    )
    _ = parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum diagram detection confidence",
    )
    args = parser.parse_args()
    images = cast(list[Path], args.images)
    confidence = cast(float, args.confidence)

    scanner = Scanner()
    results: list[SheetResult] = []
    for image in images:
        result = scanner.scan_sheet(image, confidence)
        results.append(result)
        print(f"\n{image}: {len(result.throws)} diagrams")
        print(result)

    if any(result.needs_review for result in results):
        raise LookupError(
            "Review required: some diagrams could not be classified or scored."
        )


if __name__ == "__main__":
    main()
