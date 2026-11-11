"""Shared chart parser utilities: titles, axes, ticks, labels, and legends."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from ..geometry import (
    clean_label,
    color_to_hex,
    format_number,
    group_tokens_by_row,
    is_numeric_text,
    median_or,
    row_text,
    strip_units,
)
from ..models import DataPoint, OCRToken
from ..ocr import ParsedNumber, numeric_tokens


@dataclass
class Calibration:
    slope: float
    intercept: float
    quality: float
    baseline_value: float
    tick_count: int
    source: str = "y-axis ticks"
    warning: str | None = None

    def value_at(self, pixel_y: float) -> float:
        return self.slope * pixel_y + self.intercept


@dataclass
class LegendEntry:
    label: str
    color: str
    bbox: tuple[int, int, int, int]
    confidence: float = 0.5


def find_title(
    tokens: Sequence[OCRToken],
    image_shape: tuple[int, int],
    top_fraction: float = 0.22,
) -> str | None:
    height, width = image_shape
    candidates = [token for token in tokens if token.cy < height * top_fraction and token.confidence > 0.45]
    if not candidates:
        return None
    rows = group_tokens_by_row(candidates)
    scored: list[tuple[float, str]] = []
    for row in rows:
        text = row_text(row)
        if not text or is_numeric_text(text):
            continue
        width_fraction = (max(token.x2 for token in row) - min(token.x1 for token in row)) / max(1.0, width)
        height_px = median_or([token.height for token in row])
        numeric_fraction = sum(is_numeric_text(token.text) for token in row) / max(1, len(row))
        centrality = 1.0 - min(
            1.0,
            abs(((min(token.x1 for token in row) + max(token.x2 for token in row)) / 2) - width / 2)
            / (width * 0.8),
        )
        score = (
            (0.45 + 0.55 * min(1.0, width_fraction / 0.35))
            * height_px
            * (0.7 + 0.3 * centrality)
            * (1.0 - 0.5 * numeric_fraction)
        )
        scored.append((score, text))
    if not scored:
        return None
    text = max(scored, key=lambda item: item[0])[1]
    return clean_label(text) or None


def plot_bounds_from_points(
    points: Sequence[DataPoint],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    boxes = [point.bbox for point in points if point.bbox]
    height, width = image_shape
    if not boxes:
        return (width * 0.12, height * 0.12, width * 0.95, height * 0.85)
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return (float(x1), float(y1), float(x2), float(y2))


def find_y_axis_title(
    tokens: Sequence[OCRToken],
    plot_left: float,
    image_shape: tuple[int, int],
) -> str | None:
    height, width = image_shape
    candidates = []
    for token in tokens:
        if token.x2 > plot_left + width * 0.035:
            continue
        if token.height < token.width * 1.5:
            continue
        if token.height < height * 0.025:
            continue
        if is_numeric_text(token.text):
            continue
        candidates.append(token)
    if not candidates:
        return None
    candidates.sort(key=lambda token: token.confidence * token.height, reverse=True)
    text = candidates[0].text
    if len(text) < 3:
        return None
    return clean_label(text)


def find_x_axis_title(
    tokens: Sequence[OCRToken],
    plot_bounds: tuple[float, float, float, float],
    label_tokens: Sequence[OCRToken],
    image_shape: tuple[int, int],
) -> str | None:
    plot_left, _plot_top, plot_right, plot_bottom = plot_bounds
    height, width = image_shape
    label_bottom = max((token.y2 for token in label_tokens), default=plot_bottom)
    candidates = []
    for token in tokens:
        if token.y1 < label_bottom + 1:
            continue
        if token.y1 > height * 0.96:
            continue
        if is_numeric_text(token.text):
            continue
        center = token.cx
        if not (plot_left - width * 0.15 < center < plot_right + width * 0.15):
            continue
        if token.height < 8:
            continue
        candidates.append(token)
    if not candidates:
        # Some charts put the axis title close to tick labels; look for a wide
        # non-numeric token directly under the plot.
        for token in tokens:
            if (
                token.y1 >= plot_bottom + height * 0.05
                and token.width > width * 0.08
                and not is_numeric_text(token.text)
            ):
                candidates.append(token)
    if not candidates:
        return None
    rows = group_tokens_by_row(candidates)
    rows.sort(key=lambda row: min(token.y1 for token in row))
    text = row_text(rows[0])
    return clean_label(text) or None


def _tick_candidates(
    tokens: Sequence[OCRToken],
    plot_bounds: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[list[tuple[OCRToken, ParsedNumber]], list[tuple[OCRToken, ParsedNumber]]]:
    plot_left, plot_top, plot_right, plot_bottom = plot_bounds
    height, width = image_shape
    left: list[tuple[OCRToken, ParsedNumber]] = []
    right: list[tuple[OCRToken, ParsedNumber]] = []
    for token, parsed in numeric_tokens(tokens):
        # The tallest bar can extend above the top gridline/tick label. Keep
        # enough room to include that tick while remaining near the plot.
        if token.cy < plot_top - height * 0.12 or token.cy > plot_bottom + height * 0.07:
            continue
        if token.x2 <= plot_left + width * 0.12:
            left.append((token, parsed))
        elif token.x1 >= plot_right - width * 0.12:
            right.append((token, parsed))
    return left, right


def calibrate_y_axis(
    tokens: Sequence[OCRToken],
    plot_bounds: tuple[float, float, float, float],
    baseline_y: float,
    image_shape: tuple[int, int],
    data_label_candidates: Sequence[tuple[OCRToken, ParsedNumber]] = (),
) -> Calibration | None:
    """Calibrate pixel y-coordinates to chart values using axis tick labels."""
    left, right = _tick_candidates(tokens, plot_bounds, image_shape)
    candidates = left if len(left) >= len(right) else right
    if len(candidates) >= 2:
        # Prefer tokens whose numeric content dominates the string (axis ticks),
        # then keep one token per distinct y position.
        filtered: list[tuple[OCRToken, ParsedNumber]] = []
        for token, parsed in sorted(candidates, key=lambda item: item[0].cy):
            if any(abs(token.cy - existing.cy) < 4 for existing, _ in filtered):
                continue
            filtered.append((token, parsed))
        if len(filtered) >= 2:
            pixel_y = [token.cy for token, _ in filtered]
            values = [parsed.value for _, parsed in filtered]
            from ..geometry import fit_line_robust

            fit = fit_line_robust(pixel_y, values)
            if fit is not None:
                slope, intercept, quality = fit
                baseline_value = slope * baseline_y + intercept
                warning = None
                if quality > 0.12:
                    warning = "Axis labels were noisy; values are approximate."
                if len(filtered) == 2:
                    warning = (
                        warning or "Only two axis ticks were visible, so value calibration is approximate."
                    )
                return Calibration(
                    slope=slope,
                    intercept=intercept,
                    quality=quality,
                    baseline_value=baseline_value,
                    tick_count=len(filtered),
                    source="y-axis ticks",
                    warning=warning,
                )

    # Fallback: if no axis ticks were found, infer from data labels with the
    # strongest OCR confidence. This is only used later for exact overrides.
    if data_label_candidates:
        return None
    return None


def assign_x_labels(
    bars: Sequence[DataPoint],
    tokens: Sequence[OCRToken],
    baseline_y: float,
    image_shape: tuple[int, int],
    plot_left: float,
    plot_right: float,
) -> list[str]:
    height, width = image_shape
    candidates = [
        token
        for token in tokens
        if token.y1 > baseline_y - 2
        and token.y1 < baseline_y + height * 0.32
        and token.cx > plot_left - width * 0.18
        and token.cx < plot_right + width * 0.18
        and token.confidence > 0.35
    ]
    if not candidates:
        return [point.label for point in bars]

    rows = group_tokens_by_row(candidates)
    if not rows:
        return [point.label for point in bars]
    first_row_top = min(min(token.y1 for token in row) for row in rows)
    row_height = median_or([token.height for row in rows for token in row], 12)
    band = [row for row in rows if min(token.y1 for token in row) <= first_row_top + row_height * 2.2]
    if not band:
        band = rows[:1]

    label_groups: list[tuple[float, str]] = []
    used: set[int] = set()
    for row in band:
        for token in row:
            if id(token) in used:
                continue
            # A wrapped label may continue in lower rows at the same x range.
            parts = [token]
            for other_row in rows:
                for other in other_row:
                    if other is token or id(other) in used:
                        continue
                    if other.y1 <= token.y1:
                        continue
                    if abs(other.cx - token.cx) < max(token.width, other.width) * 0.75:
                        parts.append(other)
            parts.sort(key=lambda item: (item.y1, item.x1))
            text = " ".join(part.text for part in parts)
            if is_numeric_text(text) or len(clean_label(text)) < 1:
                continue
            used.update(id(part) for part in parts)
            label_groups.append((float(np.mean([part.cx for part in parts])), clean_label(text)))

    if not label_groups:
        return [point.label for point in bars]

    bar_centers = [((point.bbox[0] + point.bbox[2]) / 2) if point.bbox else 0 for point in bars]
    gaps = np.diff(sorted(bar_centers)) if len(bar_centers) > 1 else np.array([width * 0.1])
    if len(bar_centers) >= 4 and len(gaps) >= 2:
        typical_gap = float(np.median(gaps))
        group_break = max(typical_gap * 1.6, width * 0.04)
        groups: list[list[float]] = [[]]
        for center, gap in zip(sorted(bar_centers), np.r_[0.0, gaps]):
            if groups[-1] and gap > group_break:
                groups.append([])
            groups[-1].append(center)
        if len(groups) > 1:
            inter_group_gap = max(
                (np.mean(right) - np.mean(left) for left, right in pairwise(groups)),
                default=width * 0.2,
            )
            labels_by_group: list[str] = []
            for group in groups:
                group_center = float(np.mean(group))
                nearest = min(label_groups, key=lambda item: abs(item[0] - group_center))
                labels_by_group.append(
                    nearest[1] if abs(nearest[0] - group_center) <= max(18.0, inter_group_gap * 0.45) else ""
                )
            grouped_labels: list[str] = []
            for center in bar_centers:
                group_index = min(
                    range(len(groups)),
                    key=lambda index: abs(center - float(np.mean(groups[index]))),
                )
                grouped_labels.append(labels_by_group[group_index] or f"Bar {len(grouped_labels) + 1}")
            return grouped_labels
    max_distance = max(18.0, float(np.median(gaps)) * 0.75 if len(gaps) else width * 0.12)
    labels: list[str] = []
    for point, center in zip(bars, bar_centers):
        nearest = min(label_groups, key=lambda item: abs(item[0] - center))
        if abs(nearest[0] - center) <= max_distance:
            labels.append(nearest[1])
        else:
            labels.append(point.label)
    return labels


def find_bar_value_labels(
    tokens: Sequence[OCRToken],
    point: DataPoint,
    image_shape: tuple[int, int],
) -> tuple[float, str, float] | None:
    if not point.bbox:
        return None
    x1, y1, x2, y2 = point.bbox
    height, _width = image_shape
    candidates = []
    for token, parsed in numeric_tokens(tokens):
        if token.cy < y1 - height * 0.05 or token.cy > y2 + height * 0.02:
            continue
        if token.cx < x1 - (x2 - x1) * 0.35 or token.cx > x2 + (x2 - x1) * 0.35:
            continue
        # Prefer labels above the bar, not inside it.
        if token.cy < y1 + height * 0.005:
            candidates.append((token, parsed))
    if not candidates:
        return None
    token, parsed = max(candidates, key=lambda item: item[0].confidence)
    return parsed.value, token.text, token.confidence


def find_data_label_value(
    tokens: Sequence[OCRToken],
    point: DataPoint,
    image_shape: tuple[int, int],
) -> tuple[float, str, float] | None:
    return find_bar_value_labels(tokens, point, image_shape)


def detect_legend(
    rgb: np.ndarray,
    tokens: Sequence[OCRToken],
    exclude_boxes: Sequence[tuple[int, int, int, int]] = (),
) -> list[LegendEntry]:
    """Detect colour swatches paired with nearby text labels.

    This intentionally targets compact legends used by many plotting libraries:
    a small filled rectangle followed by a text label on the same baseline.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    color_mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 45)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = rgb.shape[:2]
    entries: list[LegendEntry] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (5 <= box_width <= width * 0.12 and 4 <= box_height <= height * 0.06):
            continue
        aspect = box_width / max(1.0, box_height)
        if not (0.35 <= aspect <= 6.0):
            continue
        area = cv2.contourArea(contour)
        if area < 20 or area / max(1.0, box_width * box_height) < 0.45:
            continue
        if any(
            x >= box[0] - 3
            and y >= box[1] - 3
            and x + box_width <= box[2] + 3
            and y + box_height <= box[3] + 3
            for box in exclude_boxes
        ):
            continue
        nearby = [
            token
            for token in tokens
            if abs(token.cy - (y + box_height / 2)) < max(10, box_height * 0.9)
            and 0 <= token.x1 - (x + box_width) < width * 0.14
            and not is_numeric_text(token.text)
        ]
        if not nearby:
            continue
        label_token = min(nearby, key=lambda token: token.x1)
        crop = rgb[y : y + box_height, x : x + box_width]
        pixels = crop.reshape(-1, 3)
        if len(pixels) == 0:
            continue
        mean_bgr = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3).mean(axis=0)
        entries.append(
            LegendEntry(
                label=clean_label(label_token.text),
                color=color_to_hex(mean_bgr),
                bbox=(x, y, x + box_width, y + box_height),
                confidence=float(
                    min(0.95, (area / max(1.0, box_width * box_height)) * label_token.confidence + 0.15)
                ),
            )
        )
    # Deduplicate and keep the most confident entries.
    deduped: list[LegendEntry] = []
    for entry in sorted(entries, key=lambda item: item.confidence, reverse=True):
        if any(entry.label.lower() == existing.label.lower() for existing in deduped):
            continue
        deduped.append(entry)
    return deduped[:12]


def extract_units(axis_text: str | None, source_texts: Sequence[str] = ()) -> tuple[str | None, str | None]:
    metric = axis_text
    unit = None
    if axis_text:
        metric, unit = strip_units(axis_text)
    if unit is None:
        joined = " ".join(source_texts)
        match = re.search(
            r"(?:in|unit[s]?:?)\s+([A-Za-z%$₹€£¥][A-Za-z0-9%$\- ]{0,20})", joined, re.IGNORECASE
        )
        if match:
            unit = match.group(1).strip(" .")
    return metric, unit


def summarize_numeric_source(text: str, parsed: ParsedNumber) -> str:
    if parsed.is_percent:
        return f"{format_number(parsed.value)}%"
    return text


def line_value_from_pixel(calibration: Calibration | None, pixel_y: float) -> tuple[float | None, str]:
    if calibration is None:
        return None, "relative"
    return calibration.value_at(pixel_y), "calibrated"


def infer_number_decimals(value: float) -> int:
    if abs(value - round(value)) < 1e-6:
        return 0
    if abs(value) < 1:
        return 3
    return 2
