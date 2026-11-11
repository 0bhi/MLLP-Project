"""Turn extracted chart data into verified, human-readable insights."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from itertools import pairwise

from .geometry import format_number
from .models import ChartData, DataPoint, InsightSet

_MONTHS = {
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
}
_QUARTER_RE = re.compile(
    r"^(?:q[1-4]|(?:19|20)\d{2}(?:\s*[-/]\s*(?:q[1-4]|(?:19|20)\d{2}))?)$", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _clean_metric(chart: ChartData) -> str:
    metric = chart.metric or chart.y_label or "value"
    metric = re.sub(r"\s+", " ", str(metric)).strip(" .:")
    if metric.lower() in {"value", "values", "count", "amount"}:
        return "the metric"
    return metric


def _format_value(value: float | None, chart: ChartData, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.1f}%"
    unit = chart.unit
    if chart.chart_type == "pie":
        unit = "%"
    return format_number(value, unit)


def _is_time_series(labels: Iterable[str]) -> bool:
    labels_list = list(labels)
    if len(labels_list) < 3:
        return False
    year_hits = sum(bool(_YEAR_RE.search(label)) for label in labels_list)
    month_hits = sum(label.strip(".").lower() in _MONTHS for label in labels_list)
    quarter_hits = sum(bool(_QUARTER_RE.match(label.strip())) for label in labels_list)
    return (
        year_hits >= len(labels_list) * 0.6
        or month_hits >= len(labels_list) * 0.6
        or quarter_hits >= len(labels_list) * 0.6
    )


def _trend(points: list[DataPoint]) -> tuple[str, float | None]:
    if len(points) < 2:
        return "flat", None
    first = points[0].value
    last = points[-1].value
    if first is None or last is None:
        return "flat", None
    if abs(first) < 1e-12:
        change = None if abs(last) < 1e-12 else math.inf
    else:
        change = (last - first) / abs(first) * 100.0
    differences = [
        (later.value - earlier.value)
        for earlier, later in pairwise(points)
        if earlier.value is not None and later.value is not None
    ]
    if not differences or all(abs(diff) <= 1e-12 for diff in differences):
        direction = "flat"
    elif sum(diff > 0 for diff in differences) >= len(differences) * 0.7:
        direction = "up"
    elif sum(diff < 0 for diff in differences) >= len(differences) * 0.7:
        direction = "down"
    else:
        direction = "mixed"
    return direction, change


def _top_two_share(points: list[DataPoint], total: float) -> float | None:
    if total <= 0:
        return None
    values = sorted([point.value or 0 for point in points], reverse=True)
    return (sum(values[:2]) / total) * 100.0


def analyze_chart(chart: ChartData) -> InsightSet:
    points = [point for point in chart.points if point.value is not None]
    if not points:
        title = chart.title or "The chart"
        text = " ".join(chart.raw_text[:8]).strip()
        headline = f"{title} contains visual information, but no reliable numeric values were extracted."
        facts = [f"Recognized text: {text}"] if text else []
        if chart.warnings:
            facts.append(chart.warnings[0])
        return InsightSet(headline=headline, facts=facts)

    top = max(points, key=lambda point: point.value or -math.inf)
    bottom = min(points, key=lambda point: point.value or math.inf)
    values = [float(point.value) for point in points if point.value is not None]
    total = float(sum(values)) if all(value >= 0 for value in values) else None
    average = float(sum(values) / len(values)) if values else None
    top_share = (
        (float(top.value) / total * 100.0)
        if total and total > 0 and chart.chart_type != "pie" and top.value is not None
        else None
    )
    if chart.chart_type == "pie":
        top_share = float(top.value or 0)
    metric = _clean_metric(chart)
    top_value = _format_value(top.value, chart, percent=chart.chart_type == "pie")
    bottom_value = _format_value(bottom.value, chart, percent=chart.chart_type == "pie")
    title = chart.title or "The chart"
    facts: list[str] = []
    headline = ""

    if chart.chart_type == "pie":
        ordered_values = sorted([point.value or 0 for point in points], reverse=True)
        top_two = sum(ordered_values[:2]) if ordered_values else 0.0
        headline = (
            f"{title} shows {len(points)} shares of a whole. "
            f"{top.label} is the largest at {top_value}, while {bottom.label} is the smallest at {bottom_value}."
        )
        facts.extend(
            [
                f"{top.label} accounts for {top_value} of the total.",
                f"The top two categories together account for {top_two:.1f}%.",
                f"{bottom.label} has the smallest share at {bottom_value}.",
            ]
        )
        gap = abs((top.value or 0) - (bottom.value or 0))
        facts.append(f"The difference between the largest and smallest share is {gap:.1f} percentage points.")
    elif chart.chart_type == "infographic":
        featured = sorted(points, key=lambda point: point.confidence, reverse=True)[:3]
        headline = f"{title} highlights {len(points)} numeric facts."
        for point in featured:
            facts.append(f"{point.label}: {point.display_value or _format_value(point.value, chart)}.")
        percentages = [point for point in points if point.percentage is not None]
        if len(percentages) >= 2:
            top_percentage = max(percentages, key=lambda point: point.value or 0)
            facts.append(
                f"The highest reported percentage is {top_percentage.label} at {top_percentage.value:.0f}%."
            )
    elif chart.chart_type == "line":
        direction, change = _trend(points)
        direction_word = {"up": "an upward", "down": "a downward", "mixed": "a mixed", "flat": "a flat"}[
            direction
        ]
        headline = f"{title} shows {direction_word} trend across {len(points)} points."
        facts.append(
            f"The series starts at {_format_value(points[0].value, chart)} and ends at {_format_value(points[-1].value, chart)}."
        )
        if change is not None and math.isfinite(change):
            facts.append(f"That represents a change of {change:+.1f}% from the first to the last point.")
        facts.append(
            f"The highest point is {top.label} at {_format_value(top.value, chart)}; the lowest is {bottom.label} at {_format_value(bottom.value, chart)}."
        )
    else:
        title_text = f"{title} compares {metric} across {len(points)} categories."
        headline = (
            f"{title_text} {top.label} is highest at {top_value}; {bottom.label} is lowest at {bottom_value}."
        )
        facts.append(headline)
        if total is not None and total > 0:
            facts.append(f"The categories total {_format_value(total, chart)}.")
            if top_share is not None:
                facts.append(f"{top.label} contributes {top_share:.1f}% of the combined total.")
        if len(points) >= 2:
            gap = abs((top.value or 0) - (bottom.value or 0))
            facts.append(f"The gap between the highest and lowest category is {_format_value(gap, chart)}.")
        if average is not None and len(points) >= 4:
            facts.append(f"The average across categories is {_format_value(average, chart)}.")
        if _is_time_series([point.label for point in points]) and len(points) >= 3:
            direction, change = _trend(points)
            if direction in {"up", "down"} and change is not None and math.isfinite(change):
                facts.append(
                    f"Across the period shown, the series moves {direction} by {abs(change):.1f}% from first to last."
                )
        if top_share is not None and len(points) >= 3 and top_share >= 45:
            facts.append(f"The leading category is dominant, at {top_share:.1f}% of the total.")

    if chart.chart_type == "bar" and any(
        point.source in {"relative-bar-height", "relative"} for point in points
    ):
        facts.append(
            "Values are relative estimates from bar heights because the numeric axis could not be calibrated."
        )
    if chart.chart_type == "pie" and any(point.source == "slice angle" for point in points):
        facts.append("Some percentages were estimated from slice angles rather than printed labels.")
    if chart.confidence < 0.5:
        facts.append(
            "Extraction confidence is low; verify the reconstructed data before using these figures."
        )
    if chart.warnings:
        for warning in chart.warnings[:2]:
            if warning not in facts:
                facts.append(warning)

    return InsightSet(
        headline=headline,
        facts=list(dict.fromkeys(facts)),
        top_label=top.label,
        top_value=top.value,
        bottom_label=bottom.label,
        bottom_value=bottom.value,
        total=total,
        average=average,
        top_share=top_share,
        trend=_trend(points)[0]
        if chart.chart_type in {"line", "bar"} and _is_time_series([point.label for point in points])
        else None,
        change_percent=_trend(points)[1] if chart.chart_type in {"line", "bar"} else None,
        context={
            "metric": metric,
            "unit": chart.unit,
            "n_points": len(points),
            "chart_type": chart.chart_type,
            "sources": sorted({point.source for point in points}),
            "title": chart.title,
        },
    )
