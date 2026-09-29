"""Public result types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


def total(values: Iterable[int | None]) -> int | None:
    """Sum the values, or ``None`` when any of them is unknown."""
    summed = 0
    for value in values:
        if value is None:
            return None
        summed += value
    return summed


def _format_value(value: int | None) -> str:
    return "?" if value is None else str(value)


@dataclass
class Detection:
    """A detected pin-diagram bounding box."""

    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    column: int = -1
    row: int = -1

    @property
    def x_min(self) -> int:
        return round(self.x_center - self.width / 2)

    @property
    def y_min(self) -> int:
        return round(self.y_center - self.height / 2)

    @property
    def x_max(self) -> int:
        return round(self.x_center + self.width / 2)

    @property
    def y_max(self) -> int:
        return round(self.y_center + self.height / 2)


@dataclass(frozen=True)
class Classification:
    """The nine pin states from one diagram.

    A state is ``None`` when it is too close to the decision boundary.
    """

    pins: tuple[int | None, ...]
    confidence: float
    style: str

    @property
    def needs_review(self) -> bool:
        return None in self.pins


@dataclass
class ThrowResult:
    """Result for one pin diagram."""

    column: int
    row: int
    pins: tuple[int | None, ...]
    score: int | None
    detection_confidence: float
    classification_confidence: float

    @property
    def needs_review(self) -> bool:
        return self.score is None or None in self.pins


@dataclass
class SheetResult:
    """Result for one score sheet."""

    throws: list[ThrowResult] = field(default_factory=list)
    columns: int = 0
    rows_per_column: int = 0

    @property
    def needs_review(self) -> bool:
        return any(throw.needs_review for throw in self.throws)

    @property
    def total_pins(self) -> int | None:
        return total(throw.score for throw in self.throws)

    def __str__(self) -> str:
        """Format the detailed results and score summary."""
        lines: list[str] = []
        for throw in self.throws:
            pins = "".join(_format_value(pin) for pin in throw.pins)
            review = " REVIEW" if throw.needs_review else ""
            position = f"C{throw.column:02} R{throw.row:02}"
            lines.append(
                f"{position} {pins} => {_format_value(throw.score)} "
                f"det={throw.detection_confidence:.2f} "
                f"cls={throw.classification_confidence:.2f}{review}"
            )

        scored = [
            (throw, throw.score) for throw in self.throws if throw.score is not None
        ]
        if len(scored) != len(self.throws):
            lines.append("Zwischensummen: ? - mindestens ein Wurf ist unklar")
        else:
            by_row = self.columns > self.rows_per_column
            totals: dict[int, int] = {}
            for throw, score in scored:
                key = throw.row if by_row else throw.column
                totals[key] = totals.get(key, 0) + score

            keys = sorted(totals)
            for full_key, clearing_key in zip(keys[::2], keys[1::2]):
                full, clearing = totals[full_key], totals[clearing_key]
                label = (
                    f"Satz {full_key // 2 + 1}"
                    if by_row
                    else f"Bahn C{full_key}+C{clearing_key}"
                )
                lines.extend(
                    (
                        f"{label}: Volle={full}",
                        f"Abr={clearing} Total={full + clearing}",
                    )
                )

        lines.append(f"Total: {_format_value(self.total_pins)}")
        return "\n".join(lines)
