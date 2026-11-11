"""Interactive ChartScribe demo application."""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st
from PIL import Image

from chartscribe.pipeline import ChartScribePipeline

st.set_page_config(page_title="ChartScribe", page_icon="📊", layout="wide")
st.title("ChartScribe")
st.caption("Chart-to-text summarisation for bar charts, pie charts, line charts, and infographics")

with st.sidebar:
    chart_type = st.selectbox("Chart type", ["auto", "bar", "pie", "line", "infographic"])
    backend = st.selectbox("Summary backend", ["template", "llm"])
    st.info("Template mode is deterministic and works offline. LLM mode downloads FLAN-T5 on first use.")

image_file = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp"])

if image_file:
    image_bytes = image_file.getvalue()
    st.image(Image.open(io.BytesIO(image_bytes)), caption=image_file.name, use_container_width=True)
    if st.button("Generate summary", type="primary"):
        with st.spinner("Extracting chart content..."):
            result, debug_image = ChartScribePipeline().analyze(
                image_bytes,
                chart_type=chart_type,
                backend=backend,
                source_name=image_file.name,
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
