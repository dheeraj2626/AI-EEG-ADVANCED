from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io


def generate_pdf_report(text_report: str):
    """
    Converts a text EEG report into a PDF.
    Returns PDF as bytes (for Streamlit download).
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    for line in text_report.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return buffer
