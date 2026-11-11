"""Extractor for vertical (and simple grouped) bar charts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..classify import ClassificationResult
from ..geometry import clean_label, color_to_hex, format_number, median_or
from ..models import ChartData, DataPoint, OCRToken
from ..ocr import parse_number
from .common import (
    Calibration,
    assign_x_labels,
    calibrate_y_axis,
    detect_legend,
    extract_units,
    find_bar_value_labels,
    find_title,
    find_x_axis_title,
    find_y_axis_title,
)


@dataclass
class BarBox:
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    area: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2


class BarChartExtractor:
    def __init__(
        self, rgb: np.ndarray, tokens: Sequence[OCRToken], classification: ClassificationResult
    ) -> None:
        self.rgb = rgb
        self.tokens = list(tokens)
        self.classification = classification
        self.height, self.width = rgb.shape[:2]
        self.warnings: list[str] = []

    def _detect_bars(self) -> list[BarBox]:
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = self.height * self.width
        candidates: list[BarBox] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.0005:
                continue
            if box_width < 5 or box_height < 9:
                continue
            fill = area / max(1.0, box_width * box_height)
            if fill < 0.38:
                continue
            if box_height / max(1.0, box_width) < 0.38:
                continue
            if box_width > self.width * 0.28 or box_height > self.height * 0.85:
                continue
            patch = self.rgb[
                y + box_height // 4 : y + 3 * box_height // 4, x + box_width // 4 : x + 3 * box_width // 4
            ]
            if patch.size == 0:
                continue
            mean_rgb = patch.reshape(-1, 3).mean(axis=0)
            bgr = np.array([mean_rgb[2], mean_rgb[1], mean_rgb[0]])
            candidates.append(BarBox(x, y, x + box_width, y + box_height, color_to_hex(bgr), area))

        if not candidates:
            return []

        bottoms = [box.y2 for box in candidates]
        tolerance = max(3.0, self.height * 0.013)
        groups: list[list[BarBox]] = []
        for bottom in bottoms:
            group = [box for box, other in zip(candidates, bottoms) if abs(other - bottom) <= tolerance]
            groups.append(group)
        best_group = max(groups, key=lambda group: (len(group), sum(box.area for box in group)))
        if len(best_group) < 1:
            return []

        widths = [box.width for box in best_group]
        median_width = median_or(widths, 1.0)
        filtered = [
            box
            for box in best_group
            if 0.35 * median_width <= box.width <= 2.8 * median_width and box.height >= 8
        ]
        if len(filtered) < len(best_group):
            self.warnings.append("Some bar-like shapes were discarded as outliers.")
        filtered.sort(key=lambda box: box.x1)
        return filtered

    def _calibrate(
        self,
        plot_bounds: tuple[float, float, float, float],
        baseline_y: float,
        data_labels: list[tuple[BarBox, tuple[float, str, float]]],
    ) -> Calibration | None:
        calibration = calibrate_y_axis(
            self.tokens,
            plot_bounds,
            baseline_y,
            (self.height, self.width),
            [parse_number(value[1][1]) for value in data_labels if parse_number(value[1][1]) is not None],
        )
        if calibration is not None:
            return calibration

        # If the chart prints values above bars, fit a value-vs-height line.
        if len(data_labels) >= 2:
            heights = np.array([box.y2 - box.y1 for box, _ in data_labels], dtype=float)
            values = np.array([label[0] for _, label in data_labels], dtype=float)
            if np.ptp(heights) > 2 and np.ptp(values) > 0:
                try:
                    from sklearn.linear_model import LinearRegression

                    model = LinearRegression().fit(heights.reshape(-1, 1), values)
                    slope = float(model.coef_[0])
                    intercept = float(model.intercept_)
                    if slope > 0:
                        self.warnings.append(
                            "Values were calibrated from printed data labels rather than axis ticks."
                        )
                        return Calibration(
                            slope=-slope,
                            intercept=intercept + slope * baseline_y,
                            quality=float(model.score(heights.reshape(-1, 1), values)),
                            baseline_value=intercept,
                            tick_count=len(data_labels),
                            source="data labels",
                        )
                except (ImportError, TypeError, ValueError):
                    self.warnings.append("Printed data labels could not be used for value calibration.")

        # Last resort: use a linear value scale from 0 to the tallest bar. Values
        # are relative to the tallest bar and are explicitly marked as estimates.
        self.warnings.append("No readable numeric axis labels were found; values are relative estimates.")
        return None

    def extract(self) -> ChartData:
        bars = self._detect_bars()
        if not bars:
            return ChartData(
                chart_type="bar",
                title=find_title(self.tokens, (self.height, self.width)),
                confidence=0.2,
                warnings=["No bars could be detected. Try a clearer image."],
                extraction_method="opencv-contours",
                raw_text=[token.text for token in self.tokens],
                debug={"bars": []},
            )

        baseline_y = median_or([box.y2 for box in bars], self.height * 0.85)
        plot_bounds = (
            float(min(box.x1 for box in bars)),
            float(min(box.y1 for box in bars)),
            float(max(box.x2 for box in bars)),
            float(baseline_y),
        )
        title = find_title(self.tokens, (self.height, self.width))
        y_title = find_y_axis_title(self.tokens, plot_bounds[0], (self.height, self.width))

        provisional_points = [
            DataPoint(
                label=f"Bar {index + 1}",
                bbox=[box.x1, box.y1, box.x2, box.y2],
                color=box.color,
                confidence=0.4,
            )
            for index, box in enumerate(bars)
        ]
        data_labels: list[tuple[BarBox, tuple[float, str, float]]] = []
        for box, point in zip(bars, provisional_points):
            label_info = find_bar_value_labels(self.tokens, point, (self.height, self.width))
            if label_info is not None:
                data_labels.append((box, label_info))

        calibration = self._calibrate(plot_bounds, baseline_y, data_labels)
        label_values = {id(box): info for box, info in data_labels}

        legend_entries = detect_legend(
            self.rgb,
            self.tokens,
            exclude_boxes=[(box.x1, box.y1, box.x2, box.y2) for box in bars],
        )
        label_tokens = [
            token
            for token in self.tokens
            if not any(
                abs(token.cy - (entry.bbox[1] + entry.bbox[3]) / 2) < max(12, entry.bbox[3] - entry.bbox[1])
                and token.x1 >= entry.bbox[0] - 4
                for entry in legend_entries
            )
        ]

        # Assign x labels after bar boxes are known.
        label_names = assign_x_labels(
            provisional_points,
            label_tokens,
            baseline_y,
            (self.height, self.width),
            plot_bounds[0],
            plot_bounds[2],
        )
        x_title = find_x_axis_title(
            self.tokens,
            plot_bounds,
            [token for token in self.tokens if token.y1 > baseline_y],
            (self.height, self.width),
        )

        legend_by_color = {entry.color.lower(): entry for entry in legend_entries}

        points: list[DataPoint] = []
        for index, (box, fallback_label) in enumerate(zip(bars, label_names)):
            label = clean_label(fallback_label) or f"Bar {index + 1}"
            value: float | None = None
            display_value: str | None = None
            source = "relative-bar-height"
            confidence = 0.55

            if calibration is not None:
                value = calibration.value_at(box.y1)
                display_value = format_number(value, None)
                source = calibration.source
                confidence = 0.78 - min(0.2, calibration.quality)

            printed = label_values.get(id(box))
            if printed is not None:
                printed_value, printed_text, printed_conf = printed
                if (
                    value is None
                    or abs(value) < 1e-12
                    or abs(printed_value - value) / max(abs(printed_value), 1e-9) > 0.12
                ):
                    value = printed_value
                    source = "printed data label"
                display_value = printed_text
                confidence = max(confidence, min(0.95, printed_conf + 0.1))

            if value is None:
                # Normalise estimated bar height to a percentage of the tallest bar.
                tallest = max(1.0, max(item.height for item in bars))
                value = (box.y2 - box.y1) / tallest * 100.0
                display_value = f"~{value:.0f}% of tallest"
                source = "relative-bar-height"

            series = None
            if legend_by_color:
                nearest_legend = min(
                    legend_by_color.items(),
                    key=lambda item: _hex_distance(item[0], box.color),
                )
                if _hex_distance(nearest_legend[0], box.color) < 75:
                    series = nearest_legend[1].label

            points.append(
                DataPoint(
                    label=label,
                    value=float(value),
                    display_value=display_value,
                    color=box.color,
                    group=series,
                    confidence=float(min(0.98, max(0.2, confidence))),
                    source=source,
                    bbox=[box.x1, box.y1, box.x2, box.y2],
                )
            )

        # Detect grouped labels (multiple bars sharing one x label).
        seen: dict[str, int] = {}
        for point in points:
            seen[point.label] = seen.get(point.label, 0) + 1
        if any(count > 1 for count in seen.values()):
            self.warnings.append(
                "Multiple bars share an x-axis label; grouped/stacked series may be flattened."
            )

        if not title:
            self.warnings.append("No chart title was detected.")
        metric, unit = extract_units(y_title, [token.text for token in self.tokens])
        if not metric:
            metric = "value"
        if any(parse_number(token.text) and parse_number(token.text).is_percent for token in self.tokens):
            unit = unit or "%"

        if calibration is not None:
            if calibration.warning:
                self.warnings.append(calibration.warning)
            if abs(calibration.baseline_value) > max(1.0, abs(calibration.value_at(plot_bounds[1])) * 0.15):
                self.warnings.append(
                    "The value axis does not appear to start at zero; comparisons describe the visible range."
                )

        matched_labels = sum(1 for point in points if not point.label.startswith("Bar "))
        base_conf = 0.48
        if calibration is not None:
            base_conf += 0.18
        if matched_labels >= max(2, len(points) * 0.6):
            base_conf += 0.12
        if title:
            base_conf += 0.05
        if bars:
            base_conf += 0.08
        if any(point.source == "printed data label" for point in points):
            base_conf += 0.05
        base_conf = min(0.97, base_conf)

        return ChartData(
            chart_type="bar",
            title=title,
            x_label=x_title,
            y_label=y_title,
            metric=metric,
            unit=unit,
            points=points,
            raw_text=[token.text for token in self.tokens],
            extraction_method="opencv-color-contours",
            confidence=float(base_conf),
            warnings=list(dict.fromkeys(self.warnings)),
            debug={
                "bars": [{"bbox": [box.x1, box.y1, box.x2, box.y2], "color": box.color} for box in bars],
                "baseline_y": baseline_y,
                "plot_bounds": list(plot_bounds),
                "calibration": {
                    "slope": calibration.slope,
                    "intercept": calibration.intercept,
                    "quality": calibration.quality,
                    "source": calibration.source,
                }
                if calibration
                else None,
            },
        )


def _hex_distance(first: str, second: str) -> float:
    try:
        first_rgb = np.array([int(first[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        second_rgb = np.array([int(second[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        return float(np.linalg.norm(first_rgb - second_rgb))
    except (IndexError, TypeError, ValueError):
        return 999.0
