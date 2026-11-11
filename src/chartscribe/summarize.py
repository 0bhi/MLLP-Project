"""Grounded chart summarisation with template and local LLM backends."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import ClassVar

from .config import DEFAULT_LLM
from .geometry import format_number
from .insights import analyze_chart
from .models import ChartData, InsightSet, SummaryResult, SummaryStyle


def _fact_line(fact: str) -> str:
    return re.sub(r"\s+", " ", fact).strip(" .") + "."


def template_summary(chart: ChartData, insights: InsightSet, style: SummaryStyle = "detailed") -> str:
    """Produce a deterministic summary that is guaranteed to match extracted data."""
    facts = [_fact_line(fact) for fact in insights.facts]
    if not facts:
        if chart.raw_text:
            return (
                "The image contains text but no reliable numeric data could be extracted: "
                + ", ".join(chart.raw_text[:8])
                + "."
            )
        return "No reliable chart content could be extracted from this image."

    if style == "one_line":
        return insights.headline

    if chart.chart_type == "pie":
        top = insights.top_label or "The leading category"
        bottom = insights.bottom_label or "the smallest category"
        top_value = format_number(insights.top_value, "%")
        bottom_value = format_number(insights.bottom_value, "%")
        values = sorted([point.value or 0 for point in chart.points], reverse=True)
        top_two = sum(values[:2])
        text = (
            f"{chart.title or 'The pie chart'} shows how a whole is divided among {len(chart.points)} categories. "
            f"{top} is the largest slice at {top_value}, while {bottom} is the smallest at {bottom_value}. "
        )
        if len(values) >= 2:
            text += f"Together, the top two categories account for {top_two:.1f}% of the total. "
        text += f"The difference between the largest and smallest shares is {abs((insights.top_value or 0) - (insights.bottom_value or 0)):.1f} percentage points."
    elif chart.chart_type == "infographic":
        text = f"{chart.title or 'The infographic'} highlights {len(chart.points)} numeric facts. "
        highlights = sorted(chart.points, key=lambda point: point.confidence, reverse=True)[:3]
        rendered = [
            f"{point.label} ({point.display_value or format_number(point.value, chart.unit)})"
            for point in highlights
            if point.value is not None
        ]
        if rendered:
            text += (
                "Key figures include "
                + ", ".join(rendered[:-1])
                + (f", and {rendered[-1]}" if len(rendered) > 1 else rendered[0])
                + ". "
            )
        percentage_points = [
            point for point in chart.points if point.percentage is not None and point.value is not None
        ]
        if len(percentage_points) >= 2:
            largest_share = max(percentage_points, key=lambda point: point.value or 0)
            text += f"The highest reported percentage is {largest_share.label} at {largest_share.value:.0f}%."
    elif chart.chart_type == "line":
        direction = insights.trend or (
            "up"
            if insights.change_percent and insights.change_percent > 0
            else "down"
            if insights.change_percent
            else "flat"
        )
        points = [point for point in chart.points if point.value is not None]
        pattern = {
            "up": "an upward",
            "down": "a downward",
            "mixed": "a mixed",
            "flat": "a flat",
        }[direction]
        text = f"{chart.title or 'The line chart'} shows {pattern} pattern across {len(points)} points. "
        if points:
            text += (
                f"It starts at {format_number(points[0].value, chart.unit)} and ends at "
                f"{format_number(points[-1].value, chart.unit)}. "
            )
        if insights.change_percent is not None and math.isfinite(insights.change_percent):
            text += (
                f"This represents a {insights.change_percent:+.1f}% change from the first to the last point. "
            )
        if insights.top_label and insights.bottom_label:
            text += (
                f"The highest point is {insights.top_label} ({format_number(insights.top_value, chart.unit)}), "
                f"and the lowest is {insights.bottom_label} ({format_number(insights.bottom_value, chart.unit)})."
            )
    else:
        metric = insights.context.get("metric") or "the metric"
        title = chart.title or "The bar chart"
        text = f"{title} compares {metric} across {len(chart.points)} categories. "
        if insights.top_label and insights.bottom_label:
            text += (
                f"{insights.top_label} is highest at {format_number(insights.top_value, chart.unit)}, "
                f"while {insights.bottom_label} is lowest at {format_number(insights.bottom_value, chart.unit)}. "
            )
        if insights.total is not None and insights.total > 0:
            text += f"The categories total {format_number(insights.total, chart.unit)}. "
            if insights.top_share is not None:
                text += f"{insights.top_label} contributes {insights.top_share:.1f}% of that total. "
        if insights.average is not None and len(chart.points) >= 4:
            text += f"The average is {format_number(insights.average, chart.unit)}. "
        if insights.trend in {"up", "down"}:
            text += f"Across the period shown, the values trend {insights.trend}."
        text = text.strip()

    if style == "detailed":
        extra = [fact for fact in facts if fact.rstrip(".").lower() not in text.lower()]
        if extra:
            text += " " + " ".join(dict.fromkeys(extra))
    if chart.confidence < 0.5:
        text += " Because extraction confidence is low, verify the reconstructed values against the original image."
    return re.sub(r"\s+", " ", text).strip()


class LocalLLM:
    """Lazy CPU-friendly seq2seq summariser (FLAN-T5 by default)."""

    _instances: ClassVar[dict[str, LocalLLM]] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_LLM) -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.error: str | None = None

    @classmethod
    def get(cls, model_name: str) -> LocalLLM:
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = cls(model_name)
            return cls._instances[model_name]

    def _load(self) -> None:
        if self.model is not None or self.error is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()
            if hasattr(self.model, "to"):
                self.model.to("cpu")
            self._torch = torch
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = str(exc)
            raise RuntimeError(f"Could not load local LLM '{self.model_name}': {exc}") from exc

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        self._load()
        assert self.tokenizer is not None and self.model is not None
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with self._torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


def _numbers_in_text(text: str) -> set[str]:
    values = set()
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", text):
        normalised = match.replace(",", "").rstrip("%")
        try:
            value = float(normalised)
            values.add(f"{value:.4g}")
        except ValueError:
            continue
    return values


def _grounded(text: str, allowed_text: str) -> tuple[bool, list[str]]:
    generated_numbers = _numbers_in_text(text)
    allowed_numbers = _numbers_in_text(allowed_text)
    allowed_values = []
    for item in allowed_numbers:
        try:
            allowed_values.append(float(item))
        except ValueError:
            pass
    missing = []
    for item in generated_numbers:
        try:
            value = float(item)
        except ValueError:
            continue
        if not any(
            abs(value - candidate) <= max(0.02, abs(candidate) * 0.012) for candidate in allowed_values
        ):
            missing.append(item)
    return not missing, missing


def _llm_prompt(chart: ChartData, insights: InsightSet, draft: str, style: SummaryStyle) -> str:
    style_instruction = {
        "one_line": "Return exactly one concise sentence.",
        "short": "Return a concise paragraph of two to four sentences.",
        "detailed": "Return a detailed paragraph of at least four sentences and include every verified fact.",
    }[style]
    return (
        "Improve the grammar and flow of the chart summary below without changing its meaning, "
        "or any numbers. Do not add new facts, causes, or predictions. "
        f"{style_instruction} Return only the improved summary.\n\n"
        f"Title: {chart.title or 'Untitled'}\n"
        f"Draft summary: {draft}\n"
        "Improved summary:"
    )


def llm_summary(
    chart: ChartData,
    insights: InsightSet,
    draft: str,
    model_name: str = DEFAULT_LLM,
    style: SummaryStyle = "detailed",
) -> SummaryResult:
    """Generate a summary with a local FLAN-T5 model and verify its numbers."""
    try:
        model = LocalLLM.get(model_name)
        prompt = _llm_prompt(chart, insights, draft, style)
        generated = model.generate(prompt)
        generated = re.sub(
            r"^(Improved summary|Summary)\s*[:\-]\s*", "", generated, flags=re.IGNORECASE
        ).strip()
        if not generated:
            return SummaryResult(
                text=draft,
                style=style,
                backend="template",
                model=model_name,
                notes=["The local LLM returned empty output; template summary used instead."],
            )
        grounded, missing = _grounded(generated, draft + " " + " ".join(insights.facts))
        if not grounded:
            return SummaryResult(
                text=draft,
                style=style,
                backend="template",
                model=model_name,
                grounded=False,
                notes=[
                    f"LLM introduced unsupported number(s): {', '.join(missing[:4])}. Template summary used instead."
                ],
            )
        if len(generated) > 1400:
            generated = generated[:1400].rsplit(".", 1)[0] + "."
        return SummaryResult(
            text=generated,
            style=style,
            backend="llm",
            model=model_name,
            grounded=True,
            notes=["Summary generated locally; all numerals passed grounding checks."],
        )
    except Exception as exc:  # noqa: BLE001 - return an explicit fallback result
        return SummaryResult(
            text=draft,
            style=style,
            backend="template",
            model=model_name,
            grounded=True,
            notes=[f"Local LLM unavailable; template summary used. Reason: {str(exc)[:180]}"],
        )


@dataclass
class SummaryRequest:
    style: SummaryStyle = "detailed"
    backend: str = "template"
    model_name: str = DEFAULT_LLM


class SummaryEngine:
    def summarize(
        self, chart: ChartData, request: SummaryRequest | None = None
    ) -> tuple[SummaryResult, InsightSet]:
        request = request or SummaryRequest()
        insights = analyze_chart(chart)
        draft = template_summary(chart, insights, request.style)
        if request.backend == "llm":
            return llm_summary(chart, insights, draft, request.model_name, request.style), insights
        return SummaryResult(text=draft, style=request.style, backend="template"), insights


def facts_for_qa(chart: ChartData, insights: InsightSet) -> str:
    lines = [insights.headline]
    lines.extend(insights.facts)
    lines.append(
        "Extracted data: "
        + "; ".join(
            f"{point.label}: {point.display_value or format_number(point.value, chart.unit)}"
            for point in chart.points
            if point.value is not None
        )
    )
    return "\n".join(f"- {line}" for line in lines)
