"""Configuration and project paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"

CHART_TYPE_MODEL = MODEL_DIR / "chart_type_classifier.joblib"

MAX_IMAGE_SIDE = 1600
MIN_OCR_CONFIDENCE = 0.30
DEFAULT_LLM = os.environ.get(
    "CHARTSCRIBE_LLM",
    os.environ.get("CHARTScribe_LLM", "google/flan-t5-small"),
)

SUMMARY_STYLES = ("one_line", "short", "detailed")
CHART_TYPES = ("auto", "bar", "pie", "line", "infographic")


def ensure_directories() -> None:
    """Create runtime directories if they do not exist."""
    for path in (DATA_DIR, SAMPLE_DIR, DATASET_DIR, MODEL_DIR, DOCS_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
