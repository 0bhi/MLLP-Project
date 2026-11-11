from chartscribe.models import ChartData, DataPoint, SummaryResult
from chartscribe.summarize import SummaryEngine, SummaryRequest, llm_summary


def _chart() -> ChartData:
    return ChartData(
        chart_type="bar",
        title="Sales",
        metric="Revenue",
        points=[
            DataPoint(label="A", value=10, confidence=1.0),
            DataPoint(label="B", value=20, confidence=1.0),
        ],
        confidence=1.0,
    )


def test_template_backend_honors_all_styles() -> None:
    engine = SummaryEngine()
    for style in ("one_line", "short", "detailed"):
        result, _ = engine.summarize(_chart(), SummaryRequest(style=style, backend="template"))
        assert result.style == style
        assert result.backend == "template"
        assert result.text


def test_llm_backend_preserves_requested_style_when_model_unavailable() -> None:
    engine = SummaryEngine()
    for style in ("one_line", "short", "detailed"):
        result, _ = engine.summarize(
            _chart(),
            SummaryRequest(
                style=style,
                backend="llm",
                model_name="chartscribe-test-model-that-does-not-exist",
            ),
        )
        assert result.style == style
        assert result.text


def test_llm_one_line_result_keeps_summary_and_insights_tuple(monkeypatch) -> None:
    def fake_llm_summary(*args, **kwargs):
        return SummaryResult(text="A short grounded summary.", style="one_line", backend="llm")

    monkeypatch.setattr("chartscribe.summarize.llm_summary", fake_llm_summary)
    result, insights = SummaryEngine().summarize(
        _chart(),
        SummaryRequest(style="one_line", backend="llm"),
    )
    assert result.style == "one_line"
    assert insights.top_label == "B"


def test_llm_result_is_not_rejected_for_short_output(monkeypatch) -> None:
    class FakeModel:
        def generate(self, prompt):
            return "A short grounded summary."

    monkeypatch.setattr("chartscribe.summarize.LocalLLM.get", lambda model_name: FakeModel())
    insights = SummaryEngine().summarize(
        _chart(),
        SummaryRequest(style="detailed", backend="template"),
    )[1]
    result = llm_summary(_chart(), insights, "A detailed draft with multiple verified facts.", style="detailed")
    assert result.backend == "llm"
    assert result.text == "A short grounded summary."


def test_template_line_summary_uses_valid_wording_for_flat_and_mixed_trends() -> None:
    flat = ChartData(
        chart_type="line",
        title="Flat",
        points=[
            DataPoint(label="2020", value=10, confidence=1.0),
            DataPoint(label="2021", value=10, confidence=1.0),
            DataPoint(label="2022", value=10, confidence=1.0),
        ],
    )
    mixed = ChartData(
        chart_type="line",
        title="Mixed",
        points=[
            DataPoint(label="2020", value=10, confidence=1.0),
            DataPoint(label="2021", value=12, confidence=1.0),
            DataPoint(label="2022", value=10, confidence=1.0),
        ],
    )

    flat_text = SummaryEngine().summarize(flat)[0].text
    mixed_text = SummaryEngine().summarize(mixed)[0].text

    assert "a flat pattern" in flat_text
    assert "a mixed pattern" in mixed_text
    assert "flatward" not in flat_text
    assert "mixedward" not in mixed_text
