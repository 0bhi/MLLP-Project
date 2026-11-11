"""Extractor for single-series line charts (an optional extension)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..classify import ClassificationResult
from ..geometry import clean_label, color_to_hex, format_number, group_tokens_by_row
from ..models import ChartData, DataPoint, OCRToken
from .common import (
    calibrate_y_axis,
    find_title,
    find_x_axis_title,
    find_y_axis_title,
)


@dataclass
class LineCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    area: float


class LineChartExtractor:
    def __init__(
        self, rgb: np.ndarray, tokens: Sequence[OCRToken], classification: ClassificationResult
    ) -> None:
        self.rgb = rgb
        self.tokens = list(tokens)
        self.classification = classification
        self.height, self.width = rgb.shape[:2]
        self.warnings: list[str] = []

    def _find_line(self) -> LineCandidate | None:
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 45)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[LineCandidate] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if width < self.width * 0.2 or area < 50:
                continue
            fill = area / max(1.0, width * height)
            if fill > 0.42:
                continue
            patch = self.rgb[y : y + height, x : x + width]
            colored = patch.reshape(-1, 3)
            colored = colored[(colored.max(axis=1) > 40) & (colored.min(axis=1) < 245)]
            if len(colored) == 0:
                continue
            mean = colored.mean(axis=0)
            bgr = (mean[2], mean[1], mean[0])
            candidates.append(LineCandidate(x, y, x + width, y + height, color_to_hex(bgr), float(area)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.x2 - item.x1) * (0.6 + min(1.0, item.area / 1000)))

    def _extract_series(self, line: LineCandidate) -> list[tuple[float, float]] | None:
        patch = self.rgb[line.y1 : line.y2 + 1, line.x1 : line.x2 + 1]
        if patch.size == 0:
            return None
        target = np.array([int(line.color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        pixels = patch.astype(float)
        distances = np.linalg.norm(pixels - target.reshape(1, 1, 3), axis=2)
        mask = distances < 60
        points: list[tuple[float, float]] = []
        for local_x in range(mask.shape[1]):
            ys = np.where(mask[:, local_x])[0]
            if len(ys) == 0:
                continue
            points.append((float(local_x + line.x1), float(np.median(ys) + line.y1)))
        return points or None

    def extract(self) -> ChartData:
        line = self._find_line()
        if line is None:
            return ChartData(
                chart_type="line",
                title=find_title(self.tokens, (self.height, self.width)),
                confidence=0.2,
                warnings=["No line series could be isolated."],
                extraction_method="colour-trace",
                raw_text=[token.text for token in self.tokens],
            )
        series = self._extract_series(line)
        if not series:
            return ChartData(
                chart_type="line",
                confidence=0.2,
                warnings=["The line was detected but could not be traced."],
                extraction_method="colour-trace",
                raw_text=[token.text for token in self.tokens],
            )

        title = find_title(self.tokens, (self.height, self.width))
        plot_bounds = (float(line.x1), float(line.y1), float(line.x2), float(line.y2))
        y_title = find_y_axis_title(self.tokens, line.x1, (self.height, self.width))
        calibration = calibrate_y_axis(
            self.tokens,
            (line.x1 * 0.85, line.y1, line.x2, self.height * 0.9),
            self.height * 0.85,
            (self.height, self.width),
        )
        label_tokens = [token for token in self.tokens if token.y1 > line.y2 + 2]
        x_title = find_x_axis_title(self.tokens, plot_bounds, label_tokens, (self.height, self.width))
        labels = self._x_labels(label_tokens)
        points: list[DataPoint] = []
        samples: list[tuple[str, float, float]] = []
        if labels:
            for label_x, label_text in labels:
                nearest_x, nearest_y = min(series, key=lambda item: abs(item[0] - label_x))
                if abs(nearest_x - label_x) <= max(8.0, self.width * 0.035):
                    samples.append((label_text, nearest_x, nearest_y))
        if not samples:
            indices = np.linspace(0, len(series) - 1, min(10, len(series))).round().astype(int)
            samples = [
                (f"Point {index + 1}", series[int(index)][0], series[int(index)][1]) for index in indices
            ]

        seen_labels: set[str] = set()
        for label, pixel_x, pixel_y in samples:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            if calibration is not None:
                value = calibration.value_at(pixel_y)
                display = format_number(value)
                confidence = 0.7
                source = "y-axis calibration"
            else:
                value = None
                display = None
                confidence = 0.3
                source = "relative"
            points.append(
                DataPoint(
                    label=label,
                    value=value,
                    display_value=display,
                    color=line.color,
                    confidence=confidence,
                    source=source,
                    bbox=[int(pixel_x - 2), int(pixel_y - 2), int(pixel_x + 2), int(pixel_y + 2)],
                )
            )
        if calibration is None:
            self.warnings.append(
                "Axis calibration was unavailable, so values are expressed as relative height only."
            )
        if not labels:
            self.warnings.append("X-axis labels could not be read.")
        metric = y_title or "value"
        return ChartData(
            chart_type="line",
            title=title,
            x_label=x_title,
            y_label=y_title,
            metric=metric,
            points=points,
            raw_text=[token.text for token in self.tokens],
            extraction_method="colour-trace",
            confidence=0.5 if calibration is not None else 0.3,
            warnings=self.warnings,
            debug={"line_bbox": [line.x1, line.y1, line.x2, line.y2], "line_color": line.color},
        )

    def _x_labels(self, tokens: Sequence[OCRToken]) -> list[tuple[float, str]]:
        if not tokens:
            return []
        rows = group_tokens_by_row(tokens)
        if not rows:
            return []
        scored_rows = []
        for row in rows:
            if len(row) < 2:
                continue
            span = max(token.x2 for token in row) - min(token.x1 for token in row)
            scored_rows.append((span, min(token.y1 for token in row), row))
        if not scored_rows:
            return []
        # The x-axis tick row is the widest multi-token row below the series.
        _, _, best_row = max(scored_rows, key=lambda item: (item[0], -item[1]))
        best_row = sorted(best_row, key=lambda token: token.x1)
        labels: list[tuple[float, str]] = []
        for token in best_row:
            text = clean_label(token.text)
            if text and len(text) <= 24:
                labels.append((token.cx, text))
        labels.sort(key=lambda item: item[0])
        return labels
