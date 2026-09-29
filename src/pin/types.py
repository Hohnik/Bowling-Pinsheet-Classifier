"""Public result types."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        scores = [throw.score for throw in self.throws]
        return None if any(score is None for score in scores) else sum(scores)  # type: ignore[arg-type]
