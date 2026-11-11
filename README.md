# ChartScribe: Chart-to-Text Summarisation

ChartScribe is an end-to-end experiential learning project that reads a bar chart,
pie chart, line chart, or infographic and produces a short natural-language summary.
It combines OCR, computer-vision geometry, chart-type classification, grounded
insight generation, and an optional local LLM backend.

## Features

- OCR with RapidOCR and numeric parsing for currency, percentages, and suffixes such as `1.2M`
- Automatic chart-type detection with an explicit type override
- Extraction for bar, pie/donut, line, and infographic layouts
- Deterministic summaries that only use extracted values
- Optional local `google/flan-t5-small` summarisation with grounding checks
- Streamlit UI, CLI, debug overlays, JSON output, and sample charts

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

For a command-line run:

```powershell
chartscribe analyze data\samples\bar_revenue.png --chart-type bar --output outputs\bar.json
chartscribe analyze data\samples\pie_browser.png --chart-type pie
chartscribe batch data\samples --output outputs\sample-results.json
```

The optional LLM backend is selected with `--backend llm` or in the UI. Template
mode is the recommended reproducible mode because it requires no network access.

## Processing pipeline

```text
Image upload -> preprocessing -> OCR -> chart classification
       -> chart-specific geometry extraction -> validation/insights
       -> grounded template or local LLM summary -> JSON/UI output
```

The project intentionally keeps extracted points and the final summary together,
so every number in a generated summary can be checked against the intermediate
representation.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit demonstration interface |
| `src/chartscribe/` | Reusable library, extractors, OCR, insights, and summaries |
| `data/samples/` | Generated sample charts and their ground-truth JSON |
| `scripts/generate_data.py` | Recreates the sample chart dataset |
| `report/` | Report outline, generated PDF, and presentation materials |

## Limitations

Photographs, 3-D charts, overlapping series, handwritten labels, and very low
resolution images may reduce OCR or geometry accuracy. The warning list and
confidence value should be reported with results rather than treating extraction
as ground truth.

## Academic deliverables

The `report/` directory contains a ready-to-complete report outline, contribution
table, PDF report, and presentation deck. Regenerate the editable deliverables
with `python scripts\generate_deliverables.py`. Add group names/roll numbers,
screenshots from the app, the GitHub URL, and a recorded demonstration before
submission.

## References

- RapidOCR: https://github.com/RapidAI/RapidOCR
- OpenCV: https://opencv.org/
- Hugging Face FLAN-T5: https://huggingface.co/google/flan-t5-small
- Streamlit: https://streamlit.io/
