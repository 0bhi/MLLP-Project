"""Chart-type classification using a trained model with heuristic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CHART_TYPE_MODEL
from .geometry import clamp
from .models import ChartType, OCRToken


@dataclass
class ClassificationResult:
    chart_type: ChartType
    confidence: float
    method: str
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
        }


def _color_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 45)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _contours(mask: np.ndarray) -> list[tuple[int, int, int, int, float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        items.append((x, y, width, height, area, perimeter))
    return items


def _circle_score(rgb: np.ndarray) -> tuple[float, dict[str, float]]:
    height, width = rgb.shape[:2]
    mask = _color_mask(rgb)
    items = _contours(mask)
    image_area = height * width
    best = 0.0
    details: dict[str, float] = {}
    if not items:
        return best, details

    for x, y, box_width, box_height, area, perimeter in items:
        if area < image_area * 0.018:
            continue
        extent = area / max(1.0, box_width * box_height)
        aspect = min(box_width, box_height) / max(1.0, max(box_width, box_height))
        circularity = 4 * np.pi * area / max(1.0, perimeter * perimeter)
        circle_like = (
            0.72 * min(1.0, extent / 0.62)
            + 0.28 * max(0.0, 1.0 - abs(1.0 - aspect) * 3.0)
            + 0.20 * min(1.0, circularity / 0.8)
        )
        if aspect > 0.76 and extent > 0.48:
            best = max(best, circle_like)
            details["circle_extent"] = round(extent, 4)
            details["circle_aspect"] = round(aspect, 4)

    # Verify with Hough circles on a downsampled edge image as a second signal.
    resized = cv2.resize(rgb, (min(width, 500), int(min(width, 500) * height / width)))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, resized.shape[1] // 5),
        param1=100,
        param2=35,
        minRadius=max(20, resized.shape[1] // 12),
        maxRadius=int(resized.shape[1] * 0.48),
    )
    if circles is not None:
        largest = max(circles[0], key=lambda item: item[2])
        details["hough_radius_ratio"] = round(float(largest[2]) / resized.shape[1], 4)
        best = max(best, clamp(float(largest[2]) / resized.shape[1] * 1.5, 0, 1))
    return clamp(best, 0, 1), details


def _bar_score(rgb: np.ndarray) -> tuple[float, dict[str, float]]:
    height, width = rgb.shape[:2]
    mask = _color_mask(rgb)
    items = _contours(mask)
    image_area = height * width
    rectangles = []
    for x, y, box_width, box_height, area, _ in items:
        if (
            box_width >= 4
            and box_height >= 10
            and area >= image_area * 0.0006
            and area / max(1.0, box_width * box_height) > 0.42
            and box_height / max(1.0, box_width) > 0.45
        ):
            rectangles.append((x, y, box_width, box_height, area))

    if len(rectangles) < 2:
        return 0.0, {"bars": float(len(rectangles))}

    bottoms = [y + box_height for _, y, _, box_height, _ in rectangles]
    tolerance = max(4.0, height * 0.012)
    best_group: list[tuple[int, int, int, int, float]] = []
    for bottom in bottoms:
        group = [rect for rect, other in zip(rectangles, bottoms) if abs(other - bottom) <= tolerance]
        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < 2:
        return 0.0, {"bars": float(len(rectangles))}
    widths = [rect[2] for rect in best_group]
    lefts = [rect[0] for rect in best_group]
    spread = (max(lefts) - min(lefts)) / max(1, width)
    consistency = 1.0 - min(1.0, np.std(widths) / max(1.0, np.mean(widths))) if widths else 0
    count_score = min(1.0, len(best_group) / 5.0)
    score = 0.52 * count_score + 0.22 * consistency + 0.26 * min(1.0, spread / 0.65)
    details = {
        "bars": float(len(best_group)),
        "bar_width_cv": round(float(np.std(widths) / max(1.0, np.mean(widths))), 4),
        "bar_spread": round(float(spread), 4),
    }
    return clamp(score, 0, 1), details


def _line_score(rgb: np.ndarray) -> tuple[float, dict[str, float]]:
    height, width = rgb.shape[:2]
    mask = _color_mask(rgb)
    items = _contours(mask)
    image_area = height * width
    line_like = []
    for x, y, box_width, box_height, area, _ in items:
        if area < image_area * 0.0003:
            continue
        aspect = box_width / max(1.0, box_height)
        fill = area / max(1.0, box_width * box_height)
        if aspect > 2.2 and fill < 0.35:
            line_like.append((box_width, box_height, fill))
    if not line_like:
        return 0.0, {"line_shapes": 0.0}
    longest = max(item[0] for item in line_like)
    score = min(1.0, longest / (width * 0.55)) * min(1.0, len(line_like) / 3.0)
    return clamp(0.35 + 0.65 * score, 0, 1), {
        "line_shapes": float(len(line_like)),
        "line_span": round(longest / width, 4),
    }


def _big_numeric_tokens(tokens: list[OCRToken] | None) -> list[OCRToken]:
    if not tokens:
        return []
    heights = [token.height for token in tokens if token.height > 0]
    if not heights:
        return []
    median_height = float(np.median(heights))
    output = []
    for token in tokens:
        if token.height < median_height * 1.85:
            continue
        if not any(character.isdigit() for character in token.text):
            continue
        output.append(token)
    return output


def _infographic_score(
    rgb: np.ndarray,
    tokens: list[OCRToken] | None,
    bar: float,
    circle: float,
    line: float,
) -> float:
    height, width = rgb.shape[:2]
    if tokens:
        numeric = sum(1 for token in tokens if any(character.isdigit() for character in token.text))
        density = len(tokens) / max(1.0, (height * width) / 100_000)
        scattered = min(1.0, numeric / max(6.0, len(tokens) * 0.45))
        big_numbers = _big_numeric_tokens(tokens)
        big_signal = min(1.0, len(big_numbers) / 3.0)
        text_signal = 0.45 * min(1.0, density / 6.0) + 0.35 * scattered + 0.35 * big_signal
    else:
        text_signal = 0.3
    geometry = max(bar, circle, line)
    return clamp(0.32 + 0.45 * (1.0 - geometry) + 0.38 * text_signal, 0, 1)


def extract_classifier_features(rgb: np.ndarray) -> np.ndarray:
    """Extract a compact feature vector used by the chart-type classifier."""
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    small = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hist_h = hist_h / max(1.0, hist_h.sum())
    hist_s = hist_s / max(1.0, hist_s.sum())
    hist_v = hist_v / max(1.0, hist_v.sum())

    bar, bar_details = _bar_score(rgb)
    circle, circle_details = _circle_score(rgb)
    line, line_details = _line_score(rgb)
    mask = _color_mask(rgb)
    colored_fraction = float(mask.mean() / 255.0)
    items = _contours(mask)
    areas = np.array([item[4] for item in items], dtype=float) if items else np.array([0.0])
    boxes = np.array([[item[2], item[3]] for item in items], dtype=float) if items else np.zeros((1, 2))
    aspect = boxes[:, 0] / np.maximum(1.0, boxes[:, 1])
    features = np.concatenate(
        [
            small.flatten(),
            hist_h,
            hist_s,
            hist_v,
            np.array(
                [
                    colored_fraction,
                    len(items) / 20.0,
                    float(areas.max() / max(1.0, height * width)),
                    float(areas.sum() / max(1.0, height * width)),
                    float(np.median(aspect)) if len(aspect) else 0.0,
                    float(np.std(aspect)) if len(aspect) else 0.0,
                    bar,
                    circle,
                    line,
                    bar_details.get("bars", 0.0) / 10.0,
                    bar_details.get("bar_width_cv", 0.0),
                    bar_details.get("bar_spread", 0.0),
                    circle_details.get("circle_extent", 0.0),
                    circle_details.get("hough_radius_ratio", 0.0),
                    line_details.get("line_span", 0.0),
                ]
            ),
        ]
    )
    return features.astype(np.float32)


class ChartTypeClassifier:
    """Trained RandomForest classifier with transparent heuristic fallback."""

    def __init__(self, model_path: Path | str | None = CHART_TYPE_MODEL) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.classes_: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            return
        try:
            import joblib

            payload = joblib.load(self.model_path)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.classes_ = list(payload.get("classes", []))
            else:
                self.model = payload
                self.classes_ = list(getattr(payload, "classes_", []))
        except Exception:  # noqa: BLE001
            self.model = None

    def heuristic(self, rgb: np.ndarray, tokens: list[OCRToken] | None = None) -> ClassificationResult:
        bar, _ = _bar_score(rgb)
        circle, _ = _circle_score(rgb)
        line, _ = _line_score(rgb)
        infographic = _infographic_score(rgb, tokens, bar, circle, line)
        raw_scores = {"bar": bar, "pie": circle, "line": line, "infographic": infographic}
        big_numbers = _big_numeric_tokens(tokens)
        if len(big_numbers) >= 2 and len(tokens or []) <= 30:
            return ClassificationResult(
                chart_type="infographic",
                confidence=0.86,
                method="heuristic-layout",
                scores=raw_scores,
            )
        if bar >= 0.55 and bar >= circle * 0.98:
            winner = "bar"
        elif circle >= 0.62 and circle > bar * 0.75:
            winner = "pie"
        elif line > 0.55 and line > bar:
            winner = "line"
        else:
            winner = "infographic"
        sorted_scores = sorted(raw_scores.values(), reverse=True)
        confidence = raw_scores[winner]
        if len(sorted_scores) > 1:
            confidence = clamp(0.55 * confidence + 0.45 * (confidence - sorted_scores[1] + 0.5), 0.2, 0.98)
        return ClassificationResult(
            chart_type=winner,  # type: ignore[arg-type]
            confidence=confidence,
            method="heuristic",
            scores=raw_scores,
        )

    def predict(self, rgb: np.ndarray, tokens: list[OCRToken] | None = None) -> ClassificationResult:
        heuristic = self.heuristic(rgb, tokens)
        if self.model is None or not self.classes_:
            return heuristic
        try:
            features = extract_classifier_features(rgb).reshape(1, -1)
            probabilities = self.model.predict_proba(features)[0]
            mapping = {name: float(prob) for name, prob in zip(self.classes_, probabilities)}
            winner = max(mapping, key=mapping.get)
            confidence = mapping[winner]
            if winner not in {"bar", "pie", "line", "infographic"}:
                return heuristic
            if confidence < 0.46:
                return ClassificationResult(
                    chart_type=heuristic.chart_type,
                    confidence=max(heuristic.confidence, confidence),
                    method="heuristic-fallback",
                    scores={**heuristic.scores, **mapping},
                )
            combined = {
                name: 0.68 * mapping.get(name, 0.0) + 0.32 * heuristic.scores.get(name, 0.0)
                for name in ("bar", "pie", "line", "infographic")
            }
            winner = max(combined, key=combined.get)
            return ClassificationResult(
                chart_type=winner,  # type: ignore[arg-type]
                confidence=clamp(0.55 + 0.45 * combined[winner], 0.3, 0.99),
                method="trained-model",
                scores={**heuristic.scores, **{f"model_{k}": v for k, v in mapping.items()}},
            )
        except Exception:  # noqa: BLE001
            return heuristic
