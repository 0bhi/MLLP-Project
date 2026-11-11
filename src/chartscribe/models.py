"""Typed data structures shared by the extraction and summarisation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

try:
    import numpy as _np
except Exception:  # noqa: BLE001 - NumPy is optional
    _np = None


def json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars, tuples, and nested objects to JSON types."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if _np is not None and isinstance(value, _np.generic):
        return value.item()
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    return str(value)


ChartType = Literal["bar", "pie", "line", "infographic", "unknown"]
SummaryStyle = Literal["one_line", "short", "detailed"]


@dataclass
class OCRToken:
    text: str
    confidence: float
    box: list[list[float]]
    image_width: int = 0
    image_height: int = 0

    @property
    def x1(self) -> float:
        return min(point[0] for point in self.box)

    @property
    def y1(self) -> float:
        return min(point[1] for point in self.box)

    @property
    def x2(self) -> float:
        return max(point[0] for point in self.box)

    @property
    def y2(self) -> float:
        return max(point[1] for point in self.box)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def normalized(self) -> tuple[float, float, float, float]:
        if not self.image_width or not self.image_height:
            return (self.x1, self.y1, self.x2, self.y2)
        return (
            self.x1 / self.image_width,
            self.y1 / self.image_height,
            self.x2 / self.image_width,
            self.y2 / self.image_height,
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["center"] = [round(self.cx, 2), round(self.cy, 2)]
        return data


@dataclass
class DataPoint:
    label: str
    value: float | None = None
    display_value: str | None = None
    percentage: float | None = None
    color: str | None = None
    group: str | None = None
    confidence: float = 0.0
    source: str = "inferred"
    bbox: list[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.value is not None:
            data["value"] = round(float(self.value), 6)
        if self.percentage is not None:
            data["percentage"] = round(float(self.percentage), 3)
        if self.bbox is not None:
            data["bbox"] = [round(item) for item in self.bbox]
        return data


@dataclass
class ChartData:
    chart_type: ChartType = "unknown"
    title: str | None = None
    subtitle: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    metric: str | None = None
    unit: str | None = None
    points: list[DataPoint] = field(default_factory=list)
    raw_text: list[str] = field(default_factory=list)
    extraction_method: str = "heuristic"
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None

    @property
    def labels(self) -> list[str]:
        return [point.label for point in self.points]

    @property
    def values(self) -> list[float]:
        return [float(point.value) for point in self.points if point.value is not None]

    @property
    def valid_points(self) -> list[DataPoint]:
        return [point for point in self.points if point.value is not None]

    @property
    def total(self) -> float:
        return float(sum(self.values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "subtitle": self.subtitle,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "metric": self.metric,
            "unit": self.unit,
            "points": [point.as_dict() for point in self.points],
            "raw_text": self.raw_text,
            "extraction_method": self.extraction_method,
            "confidence": round(float(self.confidence), 4),
            "warnings": self.warnings,
            "debug": json_safe(self.debug),
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChartData:
        points = [
            DataPoint(**{k: v for k, v in item.items() if k in DataPoint.__dataclass_fields__})
            for item in data.get("points", [])
        ]
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "points"}
        return cls(points=points, **fields)


@dataclass
class InsightSet:
    headline: str
    facts: list[str]
    top_label: str | None = None
    top_value: float | None = None
    bottom_label: str | None = None
    bottom_value: float | None = None
    total: float | None = None
    average: float | None = None
    top_share: float | None = None
    trend: str | None = None
    change_percent: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class SummaryResult:
    text: str
    style: str = "detailed"
    backend: str = "template"
    model: str | None = None
    grounded: bool = True
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    chart: ChartData
    summary: SummaryResult
    insights: InsightSet
    ocr: list[OCRToken]
    elapsed_seconds: float = 0.0
    image_shape: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "chart": self.chart.as_dict(),
                "summary": self.summary.as_dict(),
                "insights": self.insights.as_dict(),
                "ocr": [token.as_dict() for token in self.ocr],
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "image_shape": self.image_shape,
            }
        )
