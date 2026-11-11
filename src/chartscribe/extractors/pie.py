"""Extractor for pie and donut charts using colour segmentation and geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..classify import ClassificationResult
from ..geometry import (
    angle_from_center,
    angular_distance,
    clean_label,
    color_to_hex,
    is_numeric_text,
)
from ..models import ChartData, DataPoint, OCRToken
from ..ocr import parse_number
from .common import detect_legend, find_title


@dataclass
class Sector:
    start_angle: float
    end_angle: float
    mid_angle: float
    share: float
    color: str
    rgb: tuple[int, int, int]
    pixel_count: int = 0


class PieChartExtractor:
    def __init__(
        self, rgb: np.ndarray, tokens: Sequence[OCRToken], classification: ClassificationResult
    ) -> None:
        self.rgb = rgb
        self.tokens = list(tokens)
        self.classification = classification
        self.height, self.width = rgb.shape[:2]
        self.warnings: list[str] = []

    def _find_circle(self) -> tuple[float, float, float] | None:
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 35)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = self.height * self.width
        best: tuple[float, float, float] | None = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.012:
                continue
            _x, _y, width, height = cv2.boundingRect(contour)
            extent = area / max(1.0, width * height)
            aspect = min(width, height) / max(1.0, max(width, height))
            if extent < 0.22 or aspect < 0.58:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            if radius < min(self.height, self.width) * 0.08:
                continue
            if area > best_area:
                best_area = area
                best = (float(center_x), float(center_y), float(radius))

        if best is not None:
            return best

        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, self.width // 5),
            param1=100,
            param2=40,
            minRadius=max(20, int(min(self.height, self.width) * 0.12)),
            maxRadius=int(min(self.height, self.width) * 0.48),
        )
        if circles is None:
            return None
        circle = max(circles[0], key=lambda item: item[2])
        return float(circle[0]), float(circle[1]), float(circle[2])

    def _palette(
        self, center_x: float, center_y: float, radius: float
    ) -> list[tuple[tuple[int, int, int], int]]:
        height, width = self.rgb.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)
        mask = (
            (distance < radius * 0.94)
            & (distance > radius * 0.28)
            & (hsv[:, :, 1] > 38)
            & (hsv[:, :, 2] > 45)
        )
        pixels = self.rgb[mask]
        if len(pixels) < 100:
            return []

        # Quantise colours and keep dense peaks; anti-aliasing produces many
        # low-count bins, while the interior of each slice produces one large bin.
        quantised = (pixels // 24) * 24
        keys, counts = np.unique(quantised, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:24]
        peaks: list[dict] = []
        for index in order:
            key = keys[index]
            count = int(counts[index])
            if count < len(pixels) * 0.008:
                continue
            matching = np.all(np.abs(pixels.astype(int) - key.astype(int)) <= 12, axis=1)
            actual = pixels[matching].mean(axis=0) if matching.any() else key
            peaks.append(
                {
                    "rgb": tuple(int(value) for value in actual),
                    "count": count,
                }
            )

        merged: list[dict] = []
        for peak in sorted(peaks, key=lambda item: item["count"], reverse=True):
            target = None
            for existing in merged:
                if (
                    np.linalg.norm(
                        np.array(peak["rgb"], dtype=float) - np.array(existing["rgb"], dtype=float)
                    )
                    < 42
                ):
                    target = existing
                    break
            if target is None:
                merged.append(dict(peak))
            else:
                total = target["count"] + peak["count"]
                target["rgb"] = tuple(
                    int(
                        (target["rgb"][channel] * target["count"] + peak["rgb"][channel] * peak["count"])
                        / max(1, total)
                    )
                    for channel in range(3)
                )
                target["count"] = total

        merged = [item for item in merged if item["count"] > len(pixels) * 0.012]
        merged = sorted(merged, key=lambda item: item["count"], reverse=True)[:9]
        return [(item["rgb"], item["count"]) for item in merged]

    def _scan_sectors(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        palette: list[tuple[tuple[int, int, int], int]],
    ) -> list[Sector]:
        step = 0.5
        angles = np.arange(0, 360.0, step)
        palette_array = np.array([item[0] for item in palette], dtype=float)
        labels = np.full(len(angles), -1, dtype=int)
        distances = np.full(len(angles), 1e9, dtype=float)
        radii = [0.55, 0.65, 0.75, 0.85]
        for index, angle in enumerate(angles):
            radians = math.radians(angle)
            best_label = -1
            best_distance = 1e9
            for fraction in radii:
                x = int(round(center_x + radius * fraction * math.cos(radians)))
                y = int(round(center_y - radius * fraction * math.sin(radians)))
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                pixel = self.rgb[y, x].astype(float)
                if np.min(pixel) > 238 or np.max(pixel) < 25:
                    continue
                color_distance = np.linalg.norm(palette_array - pixel, axis=1)
                candidate = int(np.argmin(color_distance))
                if color_distance[candidate] < best_distance and color_distance[candidate] < 95:
                    best_distance = float(color_distance[candidate])
                    best_label = candidate
            if best_label >= 0:
                labels[index] = best_label
                distances[index] = best_distance

        valid = labels >= 0
        if valid.sum() < len(labels) * 0.55:
            return []
        indices = np.arange(len(labels))
        labels[~valid] = np.interp(indices[~valid], indices[valid], labels[valid]).round().astype(int)

        # Find the first boundary so run-length encoding can wrap around.
        changes = np.where(labels != np.roll(labels, 1))[0]
        if len(changes) == 0:
            return []
        start_offset = int(changes[0])
        rolled = np.roll(labels, -start_offset)
        run_labels: list[int] = []
        run_lengths: list[int] = []
        for label in rolled:
            if run_labels and run_labels[-1] == int(label):
                run_lengths[-1] += 1
            else:
                run_labels.append(int(label))
                run_lengths.append(1)

        # Merge tiny runs (text holes, anti-aliasing) into a neighbour.
        min_run = max(2, int(4.0 / step))
        changed = True
        while changed and len(run_lengths) > 1:
            changed = False
            smallest = int(np.argmin(run_lengths))
            if run_lengths[smallest] >= min_run:
                break
            neighbor = smallest - 1 if smallest > 0 else 1
            run_lengths[neighbor] += run_lengths[smallest]
            del run_lengths[smallest]
            del run_labels[smallest]
            changed = True

        sectors: list[Sector] = []
        cursor = 0.0
        for run_label, run_length in zip(run_labels, run_lengths):
            span = run_length * step
            start = (cursor + start_offset * step) % 360.0
            end = (start + span) % 360.0
            mid = (start + span / 2.0) % 360.0
            rgb, _ = palette[run_label]
            share = span / 360.0 * 100.0
            sectors.append(
                Sector(
                    start_angle=start,
                    end_angle=end,
                    mid_angle=mid,
                    share=share,
                    color=color_to_hex((rgb[2], rgb[1], rgb[0])),
                    rgb=rgb,
                    pixel_count=run_length,
                )
            )
            cursor += span
        return sectors

    def _label_tokens(
        self, center_x: float, center_y: float, radius: float, title: str | None
    ) -> list[OCRToken]:
        output = []
        title_boxes = [token for token in self.tokens if title and token.text and token.text in (title or "")]
        legend = detect_legend(self.rgb, self.tokens)
        legend_rows = [(entry.bbox[1], entry.bbox[3], entry.bbox[2]) for entry in legend]
        for token in self.tokens:
            if token.text == title or (title and clean_label(token.text) == clean_label(title)):
                continue
            if title_boxes and any(
                abs(token.cy - box.cy) < max(token.height, box.height)
                and abs(token.cx - box.cx) < self.width * 0.4
                for box in title_boxes
            ):
                continue
            # Exclude legend labels: they sit immediately to the right of a
            # colour swatch on the same baseline.
            if any(
                top <= token.cy <= bottom + 6 and token.x1 >= right - 2 for top, bottom, right in legend_rows
            ):
                continue
            dx = token.cx - center_x
            dy = token.cy - center_y
            distance = math.hypot(dx, dy)
            ratio = distance / max(1.0, radius)
            if 0.34 <= ratio <= 1.55:
                output.append(token)
        return output

    def _label_sectors(
        self,
        sectors: list[Sector],
        center_x: float,
        center_y: float,
        radius: float,
        title: str | None,
    ) -> tuple[list[str], list[float | None], list[float]]:
        labels: list[str | None] = [None] * len(sectors)
        percentages: list[float | None] = [None] * len(sectors)
        confidences: list[float] = [0.4] * len(sectors)
        tokens = self._label_tokens(center_x, center_y, radius, title)
        percentages_by_sector: list[list[tuple[float, float, OCRToken]]] = [[] for _ in sectors]
        names_by_sector: list[list[tuple[float, float, OCRToken]]] = [[] for _ in sectors]

        for token in tokens:
            angle = angle_from_center(token.cx, token.cy, center_x, center_y)
            sector_index = min(
                range(len(sectors)),
                key=lambda index: angular_distance(angle, sectors[index].mid_angle),
            )
            delta = angular_distance(angle, sectors[sector_index].mid_angle)
            parsed = parse_number(token.text)
            if parsed is not None and (parsed.is_percent or token.text.strip().endswith("%")):
                percentages_by_sector[sector_index].append((delta, token.confidence, token))
            elif not is_numeric_text(token.text) and len(clean_label(token.text)) >= 2:
                # Labels at or outside the pie edge are candidate names.
                names_by_sector[sector_index].append((delta, token.confidence, token))

        for index, sector in enumerate(sectors):
            if percentages_by_sector[index]:
                _, confidence, token = min(percentages_by_sector[index], key=lambda item: (item[0], -item[1]))
                parsed = parse_number(token.text)
                if parsed is not None:
                    percentages[index] = float(parsed.value)
                    confidences[index] = max(confidences[index], confidence)
            if names_by_sector[index]:
                _, confidence, token = min(names_by_sector[index], key=lambda item: (item[0], -item[1]))
                labels[index] = clean_label(token.text)
                confidences[index] = max(confidences[index], confidence)

        legend = detect_legend(self.rgb, self.tokens)
        for index, sector in enumerate(sectors):
            if labels[index]:
                continue
            if legend:
                best = min(legend, key=lambda entry: _hex_distance(entry.color, sector.color))
                if _hex_distance(best.color, sector.color) < 70:
                    labels[index] = best.label
                    confidences[index] = max(confidences[index], best.confidence)
        return [label or f"Slice {index + 1}" for index, label in enumerate(labels)], percentages, confidences

    def extract(self) -> ChartData:
        title = find_title(self.tokens, (self.height, self.width))
        circle = self._find_circle()
        if circle is None:
            return ChartData(
                chart_type="pie",
                title=title,
                confidence=0.2,
                warnings=["The pie/donut circle could not be located. Try a clearer image."],
                extraction_method="colour-segmentation",
                raw_text=[token.text for token in self.tokens],
            )
        center_x, center_y, radius = circle
        palette = self._palette(center_x, center_y, radius)
        if len(palette) < 2:
            return ChartData(
                chart_type="pie",
                title=title,
                confidence=0.25,
                warnings=["Fewer than two slice colours were detected."],
                extraction_method="colour-segmentation",
                raw_text=[token.text for token in self.tokens],
            )
        sectors = self._scan_sectors(center_x, center_y, radius, palette)
        if len(sectors) < 2:
            return ChartData(
                chart_type="pie",
                title=title,
                confidence=0.25,
                warnings=["The pie slices could not be separated reliably."],
                extraction_method="colour-segmentation",
                raw_text=[token.text for token in self.tokens],
            )

        labels, percentages, label_confidences = self._label_sectors(
            sectors, center_x, center_y, radius, title
        )
        points: list[DataPoint] = []
        for index, sector in enumerate(sectors):
            percentage = percentages[index]
            value = percentage if percentage is not None else sector.share
            source = "printed percentage" if percentage is not None else "slice angle"
            confidence = 0.62 + 0.25 * label_confidences[index] if label_confidences[index] > 0.5 else 0.55
            if percentage is None:
                confidence -= 0.08
            points.append(
                DataPoint(
                    label=labels[index],
                    value=float(value),
                    display_value=f"{value:.1f}%",
                    percentage=float(value),
                    color=sector.color,
                    confidence=float(min(0.96, max(0.25, confidence))),
                    source=source,
                )
            )

        percentage_count = sum(point.source == "printed percentage" for point in points)
        if percentage_count == len(points):
            total_percentage = sum(point.value or 0 for point in points)
            if abs(total_percentage - 100) > 5:
                self.warnings.append(
                    f"Printed percentages sum to {total_percentage:.1f}%, so they may not represent a complete whole."
                )
        elif percentage_count > 0:
            self.warnings.append(
                "Only some slices had printed percentages; remaining values were estimated from angles."
            )

        label_matches = sum(not point.label.startswith("Slice ") for point in points)
        if label_matches < len(points):
            self.warnings.append("Some slice labels could not be matched; generic slice names were used.")
        if any(
            abs(sector.share - (percentages[index] or sector.share)) > 8
            for index, sector in enumerate(sectors)
        ):
            self.warnings.append("Printed percentages differ noticeably from the measured slice angles.")

        # Keep slices sorted by decreasing share for readable summaries.
        points_sorted = sorted(points, key=lambda point: point.value or 0, reverse=True)
        confidence = 0.52
        confidence += min(0.2, len(points) * 0.025)
        confidence += 0.12 if label_matches >= len(points) else 0.05
        confidence += 0.1 if percentage_count >= max(1, len(points) // 2) else 0.0
        if title:
            confidence += 0.04
        confidence = min(0.95, confidence)

        return ChartData(
            chart_type="pie",
            title=title,
            metric="share",
            unit="%",
            points=points_sorted,
            raw_text=[token.text for token in self.tokens],
            extraction_method="colour-segmentation+angular-scan",
            confidence=float(confidence),
            warnings=list(dict.fromkeys(self.warnings)),
            debug={
                "center": [center_x, center_y],
                "radius": radius,
                "palette": [color_to_hex((rgb[2], rgb[1], rgb[0])) for rgb, _ in palette],
                "sectors": [
                    {
                        "center": [center_x, center_y],
                        "radius": radius,
                        "start_angle": sector.start_angle,
                        "end_angle": sector.end_angle,
                        "mid_angle": sector.mid_angle,
                        "share": sector.share,
                        "color": sector.color,
                    }
                    for sector in sectors
                ],
            },
        )


def _hex_distance(first: str, second: str) -> float:
    try:
        first_rgb = np.array([int(first[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        second_rgb = np.array([int(second[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        return float(np.linalg.norm(first_rgb - second_rgb))
    except (IndexError, TypeError, ValueError):
        return 999.0
