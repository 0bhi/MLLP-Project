"""OCR service and numeric text parsing built on RapidOCR."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import MIN_OCR_CONFIDENCE
from .models import OCRToken


class OCRUnavailableError(RuntimeError):
    """Raised when the OCR engine cannot be initialised."""


@dataclass
class ParsedNumber:
    value: float
    text: str
    is_percent: bool = False
    is_currency: bool = False
    suffix: str | None = None
    decimals: int = 0
    negative: bool = False


_CURRENCY_RE = re.compile(r"[₹$€£¥]|(?:\b(?:rs\.?|inr|usd|eur|gbp|jpy)\b)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"%|percent", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*")
_MULTIPLIER_RE = re.compile(
    r"(?P<number>[-+]?\d+(?:\.\d+)?)\s*(?P<suffix>thousand|million|billion|trillion|k|m|bn|b|t)\b",
    re.IGNORECASE,
)
_SUFFIX_MULTIPLIERS = {
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "t": 1_000_000_000_000,
}


def _normalise_text(text: str) -> str:
    text = (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
        .replace("，", ",")
        .replace("。", ".")
        .strip()
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_numeric_text(text: str) -> str:
    text = _normalise_text(text)
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[Il|](?=\d)", "1", text)
    text = re.sub(r"(?<=\d)\s*-\s*$", "", text)
    text = re.sub(r"^[-–—]\s+(?=\d)", "-", text)
    return text.strip(" .,;:|")


def parse_number(text: str) -> ParsedNumber | None:
    """Parse an OCR string into a number, preserving percent/unit metadata.

    The parser deliberately accepts noisy tokens such as ``"50 -"``, ``"₹1,234"``,
    and ``"1.2M"`` because OCR output around chart axes is often imperfect.
    """
    cleaned = _clean_numeric_text(text)
    if not cleaned:
        return None

    is_percent = bool(_PERCENT_RE.search(cleaned))
    is_currency = bool(_CURRENCY_RE.search(cleaned))
    multiplier = 1.0
    suffix = None

    multiplier_match = _MULTIPLIER_RE.search(cleaned)
    if multiplier_match:
        suffix = multiplier_match.group("suffix").lower()
        multiplier = float(_SUFFIX_MULTIPLIERS.get(suffix, 1))
        numeric_part = multiplier_match.group("number")
    else:
        match = _NUMBER_RE.search(cleaned)
        if not match:
            return None
        numeric_part = match.group(0)

    numeric_part = numeric_part.replace(" ", "")
    # Treat comma as a thousands separator unless it is clearly a decimal comma.
    if "," in numeric_part:
        if "." in numeric_part:
            numeric_part = numeric_part.replace(",", "")
        elif re.search(r",\d{1,2}$", numeric_part) and numeric_part.count(",") == 1:
            numeric_part = numeric_part.replace(",", ".")
        else:
            numeric_part = numeric_part.replace(",", "")

    try:
        value = float(numeric_part) * multiplier
    except ValueError:
        return None

    decimals = len(numeric_part.split(".")[1]) if "." in numeric_part else 0
    return ParsedNumber(
        value=value,
        text=_normalise_text(text),
        is_percent=is_percent,
        is_currency=is_currency,
        suffix=suffix,
        decimals=decimals,
        negative=value < 0,
    )


def numeric_tokens(
    tokens: Iterable[OCRToken], min_confidence: float = MIN_OCR_CONFIDENCE
) -> list[tuple[OCRToken, ParsedNumber]]:
    """Return OCR tokens that are predominantly numeric."""
    output: list[tuple[OCRToken, ParsedNumber]] = []
    for token in tokens:
        if token.confidence < min_confidence:
            continue
        parsed = parse_number(token.text)
        if parsed is None:
            continue
        compact = re.sub(r"[\s,.:%₹$€£¥+\-–—()]", "", _normalise_text(token.text))
        compact = re.sub(r"(?i)(thousand|million|billion|trillion|k|m|bn|b|t)$", "", compact)
        if not compact or not compact[0].isdigit():
            continue
        output.append((token, parsed))
    return output


class OCRService:
    """Thread-safe lazy wrapper around RapidOCR."""

    _instance: OCRService | None = None
    _engine: object | None = None
    _lock = threading.Lock()

    def __new__(cls) -> OCRService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _get_engine(self) -> object:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
            except Exception as exc:
                raise OCRUnavailableError(
                    "RapidOCR could not be initialised. Install `rapidocr-onnxruntime`."
                ) from exc
        return self._engine

    def extract(
        self,
        bgr_image: np.ndarray,
        min_confidence: float = MIN_OCR_CONFIDENCE,
    ) -> list[OCRToken]:
        """Run OCR on a BGR image and return typed tokens."""
        engine = self._get_engine()
        try:
            result, _ = engine(bgr_image)  # type: ignore[operator]
        except Exception as exc:
            raise OCRUnavailableError(f"OCR failed: {exc}") from exc

        height, width = bgr_image.shape[:2]
        tokens: list[OCRToken] = []
        for item in result or []:
            box, text, score = item[0], item[1], float(item[2])
            if not text or score < min_confidence:
                continue
            text = _normalise_text(str(text))
            if not text:
                continue
            tokens.append(
                OCRToken(
                    text=text,
                    confidence=score,
                    box=[[float(x), float(y)] for x, y in box],
                    image_width=width,
                    image_height=height,
                )
            )
        tokens.sort(key=lambda token: (round(token.cy / 8), token.cx))
        return tokens


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    return OCRService()
