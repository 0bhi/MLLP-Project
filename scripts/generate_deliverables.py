"""Generate editable PDF and PowerPoint deliverables from the project outline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"


def build_pdf() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = [
        Paragraph("ChartScribe: Chart-to-Text Summarisation", styles["Title"]),
        Paragraph("Experiential Learning Project Report", styles["Heading2"]),
        Paragraph("Group members: [add names and roll numbers]", styles["Normal"]),
        Spacer(1, 16),
    ]
    sections = [
        (
            "1. Problem Statement and Objectives",
            "ChartScribe converts bar charts, pie charts, line charts, and infographics into structured data and concise natural-language summaries.",
        ),
        (
            "2. Methodology and Processing Pipeline",
            "Image validation and resizing -> RapidOCR -> chart classification -> chart-specific geometry extraction -> insight analysis -> grounded summary.",
        ),
        (
            "3. Models and Tools",
            "RapidOCR, OpenCV, NumPy, pandas, scikit-image, Streamlit, and optional Hugging Face google/flan-t5-small.",
        ),
        (
            "4. Implementation and Usability",
            "Run pip install -r requirements.txt, pip install -e ., then streamlit run app.py. The CLI supports single-image and batch analysis.",
        ),
        (
            "5. Results and Limitations",
            "The supplied sample charts demonstrate extraction, summaries, JSON export, and debug overlays. Small labels, blur, photographs, and unusual layouts can reduce accuracy.",
        ),
        (
            "6. Conclusion and Future Scope",
            "The prototype provides an interpretable chart-to-text pipeline. Future work includes multi-series charts, multilingual OCR, perspective correction, and larger learned datasets.",
        ),
        (
            "7. Repository and References",
            "GitHub URL: [add URL]. Video URL: [add URL]. References: RapidOCR, OpenCV, Streamlit, and Hugging Face FLAN-T5.",
        ),
    ]
    for heading, body in sections:
        story.extend(
            [Paragraph(heading, styles["Heading1"]), Paragraph(body, styles["BodyText"]), Spacer(1, 10)]
        )
    SimpleDocTemplate(str(REPORT_DIR / "ChartScribe_Report.pdf"), pagesize=A4).build(story)


def build_slides() -> None:
    from pptx import Presentation

    presentation = Presentation()
    slides = [
        ("ChartScribe", "Chart-to-text summarisation using OCR, computer vision, and LLMs"),
        (
            "Problem and objective",
            "Turn visual chart information into structured values and a short, useful explanation.",
        ),
        (
            "Processing pipeline",
            "Image -> preprocessing -> OCR -> chart classifier -> extractor -> insights -> grounded summary",
        ),
        (
            "Models and tools",
            "RapidOCR; OpenCV; NumPy; pandas; deterministic templates; optional FLAN-T5; Streamlit",
        ),
        (
            "Demonstration",
            "Upload a chart, inspect extracted points and confidence, read the summary, and download JSON.",
        ),
        (
            "Results and limitations",
            "Works on bar, pie, line, and infographic samples. Accuracy depends on image quality, layout, and readable labels.",
        ),
        (
            "Conclusion and future work",
            "Interpretable prototype today; multi-series charts, multilingual OCR, and learned chart understanding next.",
        ),
    ]
    for title, body in slides:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    presentation.save(REPORT_DIR / "ChartScribe_Presentation.pptx")


if __name__ == "__main__":
    REPORT_DIR.mkdir(exist_ok=True)
    build_pdf()
    build_slides()
    print("Generated report/ChartScribe_Report.pdf and report/ChartScribe_Presentation.pptx")
