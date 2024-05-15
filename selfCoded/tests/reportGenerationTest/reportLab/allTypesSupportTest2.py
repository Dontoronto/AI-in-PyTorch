from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np
import torch
import matplotlib.pyplot as plt
import io

class PDFReport:
    def __init__(self, filename):
        self.filename = filename
        self.elements = []
        self.doc = SimpleDocTemplate(self.filename, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.heading_style = ParagraphStyle(
            'Heading',
            parent=self.styles['Heading1'],
            fontSize=14,
            leading=16,
            spaceAfter=10,
            textColor='blue'
        )
        self.body_style = self.styles['Normal']

    def add_to_report(self, content, heading):
        # Add heading
        self.elements.append(Paragraph(heading, self.heading_style))

        # Add content
        if isinstance(content, (int, float)):
            text = str(content)
            self.elements.append(Paragraph(text, self.body_style))
        elif isinstance(content, str):
            self.elements.append(Paragraph(content, self.body_style))
        elif isinstance(content, (list, np.ndarray)):
            text = ', '.join(map(str, content))
            self.elements.append(Paragraph(text, self.body_style))
        elif isinstance(content, torch.Tensor):
            text = str(content.tolist())
            self.elements.append(Paragraph(text, self.body_style))
        elif isinstance(content, plt.Figure):
            buf = io.BytesIO()
            content.savefig(buf, format='png')
            buf.seek(0)
            self.elements.append(Image(buf))
        elif isinstance(content, bytes):
            self.elements.append(Image(io.BytesIO(content)))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # Add spacer for new line
        self.elements.append(Spacer(1, 0.2 * inch))

    def build_report(self):
        self.doc.build(self.elements)

# Example usage:
if __name__ == "__main__":
    report = PDFReport("example_report.pdf")

    # Adding different types of content to the report with headings
    report.add_to_report("This is a string", "String Heading")
    report.add_to_report(1234, "Integer Heading")
    report.add_to_report(56.78, "Float Heading")
    report.add_to_report([1, 2, 3, 4], "List Heading")
    report.add_to_report(np.array([5, 6, 7, 8]), "Numpy Array Heading")
    report.add_to_report(torch.tensor([9, 10, 11, 12]), "PyTorch Tensor Heading")

    # Adding a matplotlib figure with a heading
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    report.add_to_report(fig, "Matplotlib Figure Heading")

    # Finalize the report
    report.build_report()
