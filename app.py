"""Interactive ChartScribe demo application."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from chartscribe.config import SAMPLE_DIR
from chartscribe.pipeline import ChartScribePipeline

st.set_page_config(page_title="ChartScribe", page_icon="📊", layout="wide")
st.title("ChartScribe")
st.caption("Chart-to-text summarisation for bar charts, pie charts, line charts, and infographics")

with st.sidebar:
    chart_type = st.selectbox("Chart type", ["auto", "bar", "pie", "line", "infographic"])
    backend = st.selectbox("Summary backend", ["template", "llm"])
    st.info("Template mode is deterministic and works offline. LLM mode downloads FLAN-T5 on first use.")

sample_paths = sorted(SAMPLE_DIR.glob("*.png"))
sample_names = ["Upload your own"] + [path.stem.replace("_", " ").title() for path in sample_paths]
selected_sample = st.sidebar.selectbox("Try a sample chart", sample_names)
image_file = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp"])

sample_path: Path | None = None
if selected_sample != "Upload your own":
    sample_path = next(path for path in sample_paths if path.stem.replace("_", " ").title() == selected_sample)

if image_file is not None:
    image_bytes = image_file.getvalue()
    source_name = image_file.name
    if sample_path is not None:
        st.sidebar.info("The uploaded image is being used instead of the selected sample.")
elif sample_path is not None:
    image_bytes = sample_path.read_bytes()
    source_name = sample_path.name
    truth_path = sample_path.with_suffix(".json")
    if truth_path.exists():
        with st.expander("Sample ground truth", expanded=False):
            st.json(json.loads(truth_path.read_text(encoding="utf-8")))
else:
    image_bytes = None
    source_name = ""

if image_bytes is not None:
    st.image(Image.open(io.BytesIO(image_bytes)), caption=source_name, use_container_width=True)
    if st.button("Generate summary", type="primary"):
        with st.spinner("Extracting chart content..."):
            result, debug_image = ChartScribePipeline().analyze(
                image_bytes,
                chart_type=chart_type,
                backend=backend,
                source_name=source_name,
            )
        st.subheader("Summary")
        selected_backend = f"{backend} (effective: {result.summary.backend})"
        st.caption(f"Detailed summary · Requested backend: **{selected_backend}**")
        st.success(result.summary.text)
        if result.summary.notes:
            for note in result.summary.notes:
                if result.summary.backend != backend:
                    st.warning(note)
                else:
                    st.info(note)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted data")
            st.dataframe(
                pd.DataFrame([point.as_dict() for point in result.chart.points]), use_container_width=True
            )
            st.metric("Chart type", result.chart.chart_type)
            st.metric("Extraction confidence", f"{result.chart.confidence:.0%}")
        with col2:
            st.subheader("Processing overlay")
            st.image(debug_image, channels="BGR", use_container_width=True)
            st.download_button(
                "Download JSON result",
                json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
                file_name="chartscribe-result.json",
                mime="application/json",
            )
        if result.chart.warnings:
            st.warning(" ".join(result.chart.warnings))
