from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
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

    def add_to_report(self, content):
        if isinstance(content, (int, float)):
            text = str(content)
            self.elements.append(Paragraph(text, self.styles['Normal']))
        elif isinstance(content, str):
            self.elements.append(Paragraph(content, self.styles['Normal']))
        elif isinstance(content, (list, np.ndarray)):
            text = ', '.join(map(str, content))
            self.elements.append(Paragraph(text, self.styles['Normal']))
        elif isinstance(content, torch.Tensor):
            text = str(content.tolist())
            self.elements.append(Paragraph(text, self.styles['Normal']))
        elif isinstance(content, plt.Figure):
            buf = io.BytesIO()
            content.savefig(buf, format='png')
            buf.seek(0)
            self.elements.append(Image(buf))
        elif isinstance(content, bytes):
            self.elements.append(Image(io.BytesIO(content)))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def build_report(self):
        self.doc.build(self.elements)

# Example usage:
if __name__ == "__main__":
    report = PDFReport("example_report.pdf")

    # Adding different types of content to the report
    report.add_to_report("This is a string")
    report.add_to_report(1234)
    report.add_to_report(56.78)
    report.add_to_report([1, 2, 3, 4])
    report.add_to_report(np.array([5, 6, 7, 8]))
    report.add_to_report(torch.tensor([9, 10, 11, 12]))

    # Adding a matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    report.add_to_report(fig)

    # Finalize the report
    report.build_report()