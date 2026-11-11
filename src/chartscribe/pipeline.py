"""End-to-end chart-to-text pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .classify import ChartTypeClassifier
from .config import DEFAULT_LLM
from .extractors import BarChartExtractor, InfographicExtractor, LineChartExtractor, PieChartExtractor
from .insights import analyze_chart
from .models import ChartData, PipelineResult, SummaryStyle
from .ocr import OCRService, get_ocr_service
from .preprocess import basic_quality_report, load_image
from .summarize import SummaryEngine, SummaryRequest

logger = logging.getLogger(__name__)


class ChartScribePipeline:
    def __init__(
        self,
        classifier: ChartTypeClassifier | None = None,
        ocr_service: OCRService | None = None,
        summary_engine: SummaryEngine | None = None,
    ) -> None:
        self.classifier = classifier or ChartTypeClassifier()
        self.ocr_service = ocr_service or get_ocr_service()
        self.summary_engine = summary_engine or SummaryEngine()

    def analyze(
        self,
        image: str | Path | bytes | np.ndarray,
        chart_type: str = "auto",
        style: SummaryStyle = "detailed",
        backend: str = "template",
        model_name: str = DEFAULT_LLM,
        source_name: str | None = None,
        min_ocr_confidence: float = 0.3,
    ) -> tuple[PipelineResult, np.ndarray]:
        """Run OCR, chart classification, extraction, and summarisation."""
        start = time.perf_counter()
        rgb, bgr = load_image(image)
        quality = basic_quality_report(rgb)
        tokens = self.ocr_service.extract(bgr, min_confidence=min_ocr_confidence)
        classification = self.classifier.predict(rgb, tokens)
        resolved_type = classification.chart_type if chart_type == "auto" else chart_type

        extractor_map = {
            "bar": BarChartExtractor,
            "pie": PieChartExtractor,
            "line": LineChartExtractor,
            "infographic": InfographicExtractor,
        }
        if resolved_type == "unknown":
            resolved_type = "infographic"
        if resolved_type not in extractor_map:
            raise ValueError(f"Unsupported chart type: {resolved_type}")

        extractor = extractor_map[resolved_type](rgb, tokens, classification)
        chart = extractor.extract()
        if chart_type == "auto" and chart.chart_type == "unknown":
            chart.chart_type = resolved_type  # type: ignore[assignment]
        chart.source_file = source_name
        chart.raw_text = chart.raw_text or [token.text for token in tokens]
        chart.debug["classification"] = classification.as_dict()
        chart.debug["image_quality"] = quality

        if quality.get("is_blurry"):
            chart.warnings.append("The image appears blurry, which may reduce OCR and extraction accuracy.")
        if not tokens:
            chart.warnings.append("No text was recognized in the image.")
        chart.warnings = list(dict.fromkeys(chart.warnings))

        request = SummaryRequest(style=style, backend=backend, model_name=model_name)
        summary, insights = self.summary_engine.summarize(chart, request)
        elapsed = time.perf_counter() - start
        result = PipelineResult(
            chart=chart,
            summary=summary,
            insights=insights,
            ocr=tokens,
            elapsed_seconds=elapsed,
            image_shape=(int(rgb.shape[1]), int(rgb.shape[0])),
        )
        from .preprocess import draw_debug_overlay

        debug_image = draw_debug_overlay(rgb, tokens, chart)
        return result, debug_image

    def summarize_existing(
        self,
        chart: ChartData,
        style: SummaryStyle = "detailed",
        backend: str = "template",
        model_name: str = DEFAULT_LLM,
    ) -> tuple[Any, Any]:
        summary, insights = self.summary_engine.summarize(
            chart,
            SummaryRequest(style=style, backend=backend, model_name=model_name),
        )
        return summary, insights

    @staticmethod
    def answer_question(
        question: str,
        chart: ChartData,
        backend: str = "template",
        model_name: str = DEFAULT_LLM,
    ) -> str:
        """Answer common factual questions using extracted data (grounded)."""
        from .geometry import format_number

        insights = analyze_chart(chart)
        q = question.lower()
        if not question.strip():
            return "Ask a question about the extracted chart data."
        points = [point for point in chart.points if point.value is not None]
        if not points:
            if backend == "llm":
                from .summarize import LocalLLM

                prompt = (
                    "Answer the question using only the OCR text below. If the answer is not present, say you cannot determine it.\n"
                    f"OCR text: {', '.join(chart.raw_text[:20])}\nQuestion: {question}\nAnswer:"
                )
                try:
                    return LocalLLM.get(model_name).generate(prompt, max_new_tokens=80)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("LLM question answering was unavailable: %s", exc)
            return "The chart data needed to answer that question was not extracted reliably."
        top = max(points, key=lambda point: point.value or 0)
        bottom = min(points, key=lambda point: point.value or 0)
        if any(word in q for word in ("highest", "largest", "most", "maximum", "max", "top")):
            return f"{top.label} is the highest at {format_number(top.value, chart.unit)}."
        if any(word in q for word in ("lowest", "smallest", "least", "minimum", "min", "bottom")):
            return f"{bottom.label} is the lowest at {format_number(bottom.value, chart.unit)}."
        if any(word in q for word in ("total", "sum", "altogether")):
            if chart.chart_type == "pie":
                return "The displayed shares sum to " + f"{sum(point.value or 0 for point in points):.1f}%."
            return f"The total across all categories is {format_number(sum(point.value or 0 for point in points), chart.unit)}."
        if any(word in q for word in ("average", "mean")):
            return f"The average is {format_number(sum(point.value or 0 for point in points) / len(points), chart.unit)}."
        if any(word in q for word in ("difference", "gap", "compare")):
            gap = abs((top.value or 0) - (bottom.value or 0))
            return (
                f"{top.label} is {format_number(gap, chart.unit)} above {bottom.label} "
                f"({format_number(top.value, chart.unit)} versus {format_number(bottom.value, chart.unit)})."
            )
        if any(word in q for word in ("share", "percent", "proportion", "fraction")):
            total = sum(point.value or 0 for point in points)
            share = (top.value or 0) / total * 100 if total else 0
            return f"{top.label} accounts for {share:.1f}% of the total shown."
        if any(word in q for word in ("trend", "increase", "decrease", "grow")):
            change = insights.change_percent
            direction = insights.trend or "mixed"
            if change is not None:
                return f"The series trends {direction}, with a change of {change:+.1f}% from first to last."
            return f"The visible series is broadly {direction}."
        if any(word in q for word in ("how many", "count", "number of")):
            return f"The chart contains {len(points)} extracted categories/points."
        if backend == "llm":
            from .summarize import LocalLLM, facts_for_qa

            prompt = (
                "Answer the question using only these verified chart facts. If the answer is not present, say so.\n"
                f"Facts:\n{facts_for_qa(chart, insights)}\nQuestion: {question}\nAnswer:"
            )
            try:
                return LocalLLM.get(model_name).generate(prompt, max_new_tokens=90)
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM question answering was unavailable: %s", exc)
        return (
            "I can answer questions about the highest, lowest, total, average, gap, share, trend, "
            "or number of categories in the extracted data."
        )
