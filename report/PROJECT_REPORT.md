# ChartScribe Project Report

> Replace bracketed fields with the group's names, roll numbers, contribution
> details, GitHub URL, screenshots, and presentation-video link.

## 1. Title and Group Details

**Title:** ChartScribe: Chart-to-Text Summarisation Using Computer Vision, OCR, and LLMs  
**Members:** [Name — Roll number]  
**Contributions:** [Member 1], [Member 2], [Member 3], [Member 4]

## 2. Problem Statement and Objectives

Charts communicate quantitative information compactly, but users must visually
decode labels, values, and comparisons. ChartScribe converts a chart image into
structured data and a concise natural-language explanation.

Objectives are to classify the visual, recover labels and values, calculate
high-level insights, produce a grounded summary, and expose the process through
an accessible prototype.

## 3. Proposed Methodology and Processing Pipeline

```text
Input image -> image validation/resizing -> RapidOCR
    -> chart-type classifier -> bar/pie/line/infographic extractor
    -> numeric validation and insight analysis -> summary generation -> output
```

OCR tokens retain text, confidence, and bounding boxes. Extractors use color
regions, contours, circles, lines, spatial grouping, and OCR positions. The
insight layer calculates extrema, totals, averages, shares, and trends. The
template backend is deterministic; the optional FLAN-T5 backend is checked for
numeric grounding before it is returned.

## 4. Models and Tools Used

| Component | Tool/model | Purpose |
|---|---|---|
| OCR | RapidOCR ONNX Runtime | Text and numeric token extraction |
| Vision | OpenCV, NumPy, scikit-image | Segmentation, contours, circles, lines |
| Data handling | pandas | Extracted-data tables in the UI |
| Summary | Grounded templates | Reproducible offline output |
| Optional LLM | [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) | Natural-language rewriting |
| Interface | Streamlit | Demonstration application |

## 5. Implementation and Usability

Install dependencies from `requirements.txt`, install the package with
`pip install -e .`, and run `streamlit run app.py`. The CLI also supports
single-image analysis, batch analysis, JSON export, and debug overlays.

## 6. Results and Sample Outputs

Use the supplied images in `data/samples/`. Record the original image, extracted
points, generated summary, confidence, runtime, and warnings for at least one bar
chart and one pie chart. Discuss errors caused by small labels, legends, blur,
and unusual layouts.

## 7. Conclusion and Future Scope

The prototype demonstrates an interpretable chart-to-text pipeline rather than a
black-box response: extracted points and intermediate confidence values are
visible before summarisation. Future work includes multi-series charts, better
perspective correction, chart datasets for learned classification, multilingual
OCR, and accessibility integrations.

## 8. References and GitHub Repository Link

**Repository:** [Insert GitHub URL]  
**Video:** [Insert accessible video URL]  
References are listed in the repository README and include RapidOCR, OpenCV,
Streamlit, and Hugging Face FLAN-T5.
