"""Extractor for infographics, stat cards, and text-heavy metric panels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..classify import ClassificationResult
from ..geometry import clean_label, group_tokens_by_row, is_numeric_text, median_or, row_text
from ..models import ChartData, DataPoint, OCRToken
from ..ocr import numeric_tokens
from .common import find_title


@dataclass
class MetricCandidate:
    label: str
    value: float
    raw: str
    confidence: float
    y: float
    x: float
    is_percent: bool = False


class InfographicExtractor:
    def __init__(
        self, rgb: np.ndarray, tokens: Sequence[OCRToken], classification: ClassificationResult
    ) -> None:
        self.rgb = rgb
        self.tokens = list(tokens)
        self.classification = classification
        self.height, self.width = rgb.shape[:2]
        self.warnings: list[str] = []

    def _row_label(self, row: list[OCRToken], numeric_token: OCRToken) -> str:
        by_side_left = [
            token for token in row if token.x2 <= numeric_token.x1 - 2 and not is_numeric_text(token.text)
        ]
        by_side_right = [
            token for token in row if token.x1 >= numeric_token.x2 + 2 and not is_numeric_text(token.text)
        ]
        candidates = by_side_right or by_side_left
        if candidates:
            candidates.sort(key=lambda token: abs(token.cx - numeric_token.cx))
            text = " ".join(token.text for token in sorted(candidates, key=lambda token: token.x1))
            cleaned = clean_label(text)
            if cleaned and not re.fullmatch(r"[-–—•\s]+", cleaned):
                return cleaned
        return ""

    def _nearby_label(self, numeric_token: OCRToken, rows: list[list[OCRToken]]) -> str:
        best: tuple[float, str] | None = None
        for row in rows:
            if any(token is numeric_token for token in row):
                continue
            text = row_text(row)
            if not text or is_numeric_text(text):
                continue
            row_y = median_or([token.cy for token in row], 0.0)
            vertical = abs(row_y - numeric_token.cy)
            if vertical > self.height * 0.16:
                continue
            # Prefer the token in the row that is horizontally closest to the
            # big number, then include adjacent tokens to complete the label.
            closest = min(row, key=lambda token: abs(token.cx - numeric_token.cx))
            horizontal = abs(closest.cx - numeric_token.cx)
            if horizontal > self.width * 0.18:
                continue
            nearby_tokens = [
                token
                for token in row
                if abs(token.cx - closest.cx) < max(token.width, closest.width) * 1.6
                or (token.x1 <= closest.x2 and token.x2 >= closest.x1)
            ]
            label = clean_label(
                " ".join(token.text for token in sorted(nearby_tokens, key=lambda item: item.x1))
            )
            score = vertical + horizontal * 0.6
            if label and (best is None or score < best[0]):
                best = (score, label)
        return best[1] if best else ""

    def extract(self) -> ChartData:
        title = find_title(self.tokens, (self.height, self.width), top_fraction=0.3)
        rows = group_tokens_by_row(self.tokens)
        numbers = numeric_tokens(self.tokens)
        metrics: list[MetricCandidate] = []
        seen: set[tuple[float, str]] = set()

        for token, parsed in numbers:
            label = ""
            for row in rows:
                if any(item is token for item in row):
                    label = self._row_label(row, token)
                    break
            if not label:
                label = self._nearby_label(token, rows)
            if not label:
                label = "metric"
            label = clean_label(label)
            key = (round(float(parsed.value), 6), label.lower())
            if key in seen:
                continue
            seen.add(key)
            confidence = min(0.97, token.confidence + (0.08 if label != "metric" else -0.12))
            metrics.append(
                MetricCandidate(
                    label=label,
                    value=float(parsed.value),
                    raw=token.text,
                    confidence=float(confidence),
                    y=token.cy,
                    x=token.cx,
                    is_percent=parsed.is_percent,
                )
            )

        metrics.sort(key=lambda metric: (-metric.confidence, metric.y, metric.x))
        if len(metrics) > 8:
            self.warnings.append("Many numeric values were detected; only the eight most salient are shown.")
            metrics = metrics[:8]

        points = [
            DataPoint(
                label=metric.label,
                value=metric.value,
                display_value=metric.raw,
                percentage=metric.value if metric.is_percent else None,
                confidence=metric.confidence,
                source="infographic text",
            )
            for metric in metrics
        ]

        text_rows = [row_text(row) for row in rows]
        text_rows = [text for text in text_rows if text and len(text) > 2]
        key_texts = [text for text in text_rows if not is_numeric_text(text)]
        if not points:
            self.warnings.append(
                "No numeric metrics were detected; the summary is based on the visible text."
            )
        if len(points) < 2:
            self.warnings.append("Only one numeric metric was detected.")
        if len(self.tokens) < 3:
            self.warnings.append("Very little text was recognized in the infographic.")

        confidence = 0.45 + min(0.28, len(points) * 0.04) + (0.12 if title else 0)
        confidence += (
            min(0.1, np.mean([token.confidence for token in self.tokens]) * 0.1) if self.tokens else 0
        )
        return ChartData(
            chart_type="infographic",
            title=title,
            metric="reported metrics",
            points=points,
            raw_text=[token.text for token in self.tokens],
            extraction_method="layout-aware-ocr",
            confidence=float(min(0.92, confidence)),
            warnings=list(dict.fromkeys(self.warnings)),
            debug={"text_rows": text_rows[:30], "key_texts": key_texts[:12]},
        )
