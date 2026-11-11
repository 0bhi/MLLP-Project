"""Small geometry and text-grouping utilities used by the extractors."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from .models import OCRToken


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def median_or(values: Sequence[float], fallback: float = 0.0) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else fallback


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def angular_distance(a: float, b: float) -> float:
    difference = abs((a - b) % 360.0)
    return min(difference, 360.0 - difference)


def angle_from_center(cx: float, cy: float, center_x: float, center_y: float) -> float:
    """Return an angle in degrees with 0 at east and increasing counter-clockwise."""
    return math.degrees(math.atan2(-(cy - center_y), cx - center_x)) % 360.0


def group_tokens_by_row(tokens: Iterable[OCRToken], overlap_ratio: float = 0.35) -> list[list[OCRToken]]:
    """Group OCR tokens that share a visual line."""
    rows: list[list[OCRToken]] = []
    for token in sorted(tokens, key=lambda item: (item.cy, item.cx)):
        placed = False
        for row in rows:
            row_y1 = min(item.y1 for item in row)
            row_y2 = max(item.y2 for item in row)
            row_height = max(1.0, row_y2 - row_y1)
            overlap = min(row_y2, token.y2) - max(row_y1, token.y1)
            if overlap > overlap_ratio * min(row_height, max(1.0, token.height)):
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda item: item.x1)
    rows.sort(key=lambda row: (min(item.y1 for item in row), min(item.x1 for item in row)))
    return rows


def row_text(row: Iterable[OCRToken]) -> str:
    return " ".join(token.text for token in sorted(row, key=lambda item: item.x1)).strip()


def clean_label(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip(" .|:;-")
    text = re.sub(r"^[\-\u2022\u25cf\u25aa]+\s*", "", text)
    return text.strip()


def is_numeric_text(text: str) -> bool:
    compact = re.sub(r"[\s,.:%₹$€£¥+\-–—()]", "", text)
    compact = re.sub(r"(?i)(thousand|million|billion|trillion|k|m|bn|b|t)$", "", compact)
    return bool(compact) and compact[0].isdigit()


def fit_line_robust(pixel_y: list[float], values: list[float]) -> tuple[float, float, float] | None:
    """Fit ``value = slope * y + intercept`` robustly.

    Returns ``(slope, intercept, quality)`` where quality is the median absolute
    relative residual. Theil-Sen is used when scikit-learn is available, with a
    numpy fallback.
    """
    if len(pixel_y) < 2:
        return None
    x = np.asarray(pixel_y, dtype=float)
    y = np.asarray(values, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    try:
        from sklearn.linear_model import TheilSenRegressor

        model = TheilSenRegressor(random_state=42, max_subpopulation=2000)
        model.fit(x.reshape(-1, 1), y)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
    except (ImportError, TypeError, ValueError):
        slope, intercept = np.polyfit(x, y, 1)
        slope, intercept = float(slope), float(intercept)

    predicted = slope * x + intercept
    scale = np.maximum(np.abs(y), 1e-9)
    residual = float(np.median(np.abs(predicted - y) / scale))
    if slope >= 0:
        # Y pixels increase downward, so a valid value axis must have a negative slope.
        return None
    return slope, intercept, residual


def color_to_hex(bgr: Sequence[int] | np.ndarray) -> str:
    b, g, r = [int(clamp(float(value), 0, 255)) for value in bgr[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def strip_units(text: str) -> tuple[str, str | None]:
    """Split a label like ``Revenue (USD billions)`` into name and unit."""
    match = re.search(r"\(([^()]{1,36})\)\s*$", text)
    if match:
        unit = match.group(1).strip()
        name = text[: match.start()].strip()
        return name or text, unit
    return text, None


def format_number(value: float | None, unit: str | None = None, decimals: int | None = None) -> str:
    if value is None:
        return "n/a"
    absolute = abs(value)
    if decimals is None:
        if absolute >= 1000:
            decimals = 0
        elif absolute >= 100:
            decimals = 1
        elif absolute >= 10 or absolute >= 1:
            decimals = 2
        else:
            decimals = 3
    if absolute >= 1_000_000_000:
        text = f"{value / 1_000_000_000:.2f}B"
    elif absolute >= 1_000_000:
        text = f"{value / 1_000_000:.2f}M"
    elif absolute >= 10_000:
        text = f"{value:,.0f}"
    else:
        text = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
    if unit:
        if unit.startswith("%"):
            return f"{text}%"
        if unit.startswith(("₹", "$", "€", "£", "¥")):
            return f"{unit}{text}"
        return f"{text} {unit}"
    return text


def cluster_indices(values: Sequence[float], tolerance: float) -> list[list[int]]:
    """Single-linkage style grouping of integer indices by value tolerance."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    groups: list[list[int]] = [[order[0]]]
    for index in order[1:]:
        if abs(values[index] - values[groups[-1][-1]]) <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def most_common(items: Sequence, default=None):
    if not items:
        return default
    counts: dict = defaultdict(int)
    for item in items:
        counts[item] += 1
    return max(counts.items(), key=lambda pair: pair[1])[0]
