"""Image loading, validation, and visual debug helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MAX_IMAGE_SIDE


class ImageLoadError(ValueError):
    """Raised when an image cannot be decoded."""


def load_image(source: str | Path | bytes | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return ``(rgb, bgr)`` arrays.

    The function accepts a path, raw bytes, or a BGR/RGB numpy array. Images are
    normalised to EXIF orientation and capped at ``MAX_IMAGE_SIDE`` on the longest
    side while preserving aspect ratio.
    """
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        rgb = array
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ImageLoadError(f"Image not found: {path}")
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ImageLoadError(f"Could not decode image: {path}")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(source, bytes):
        data = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ImageLoadError("Could not decode image bytes.")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"Unsupported image source: {type(source)!r}")

    pil = Image.fromarray(rgb)
    pil = ImageOps.exif_transpose(pil)
    rgb = np.asarray(pil.convert("RGB"))

    height, width = rgb.shape[:2]
    if max(height, width) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(height, width)
        rgb = cv2.resize(
            rgb,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(bgr)


def basic_quality_report(rgb: np.ndarray) -> dict[str, Any]:
    """Return simple image-quality diagnostics used by the UI and reports."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = gray.shape
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    colored_fraction = float(np.mean((hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 40)))
    white_fraction = float(np.mean(np.all(rgb > 235, axis=2)))
    return {
        "width": int(width),
        "height": int(height),
        "blur_score": round(blur_score, 2),
        "is_blurry": blur_score < 45.0,
        "colored_fraction": round(colored_fraction, 4),
        "white_fraction": round(white_fraction, 4),
    }


def draw_debug_overlay(
    rgb: np.ndarray,
    tokens: list[Any],
    chart: Any | None = None,
) -> np.ndarray:
    """Draw OCR boxes, extracted bars/slices, and labels for debugging."""
    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()

    for token in tokens:
        points = np.array(token.box, dtype=np.int32)
        color = (22, 119, 255)
        if token.confidence < 0.55:
            color = (255, 145, 30)
        cv2.polylines(overlay, [points], True, color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"{token.text[:28]} {token.confidence:.2f}",
            (int(token.x1), max(12, int(token.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    if chart is not None:
        for index, point in enumerate(getattr(chart, "points", []), start=1):
            if point.bbox:
                x1, y1, x2, y2 = [int(round(item)) for item in point.bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (34, 177, 76), 2)
                label = point.label or str(index)
                if len(label) > 18:
                    label = label[:17] + "..."
                value = point.display_value or (f"{point.value:g}" if point.value is not None else "")
                cv2.putText(
                    overlay,
                    f"{index}. {label} {value}".strip(),
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 90, 40),
                    1,
                    cv2.LINE_AA,
                )
        debug_meta = getattr(chart, "debug", {}) or {}
        for item in debug_meta.get("sectors", []):
            center = tuple(int(round(value)) for value in item.get("center", (0, 0)))
            radius = int(round(item.get("radius", 0)))
            if radius > 0:
                cv2.circle(overlay, center, radius, (34, 177, 76), 2)
                angle = float(item.get("mid_angle", 0.0))
                end = (
                    int(round(center[0] + radius * np.cos(np.deg2rad(angle)))),
                    int(round(center[1] - radius * np.sin(np.deg2rad(angle)))),
                )
                cv2.line(overlay, center, end, (34, 177, 76), 2, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def flatten_on_white(image: np.ndarray) -> np.ndarray:
    """Convert transparent images to a white RGB background."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    return image
