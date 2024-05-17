from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
import tensorFormatter
import numpyArrayFormatter


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
        elif isinstance(content, str) and content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            self.elements.append(Image(content))
        elif isinstance(content, str):
            self.elements.append(Paragraph(content, self.body_style))
        elif isinstance(content, list):
            text = ', '.join(map(str, content))
            self.elements.append(Paragraph(text, self.body_style))
        elif isinstance(content, np.ndarray):
            text = numpyArrayFormatter.NumpyArrayFormatter.format_array(content)
            self.elements.append(Preformatted(text, self.body_style))
        elif isinstance(content, torch.Tensor):
            text = tensorFormatter.TensorFormatter.format_tensor(content)
            self.elements.append(Preformatted(text, self.body_style))
        elif isinstance(content, plt.Figure):
            buf = io.BytesIO()
            content.savefig(buf, format='png')
            buf.seek(0)
            self.elements.append(Image(buf))
        elif isinstance(content, bytes):
            self.elements.append(Image(io.BytesIO(content)))
        elif isinstance(content, PILImage.Image):
            buf = io.BytesIO()
            content.save(buf, format='PNG')
            buf.seek(0)
            self.elements.append(Image(buf))
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # Add spacer for new line
        self.elements.append(Spacer(1, 0.2 * inch))

    def _create_element(self, content):
        if isinstance(content, (int, float)):
            return Paragraph(str(content), self.body_style)
        elif isinstance(content, str) and content.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return Image(content)
        elif isinstance(content, str):
            return Paragraph(content, self.body_style)
        elif isinstance(content, list):
            text = ', '.join(map(str, content))
            self.elements.append(Paragraph(text, self.body_style))
        elif isinstance(content, np.ndarray):
            text = numpyArrayFormatter.NumpyArrayFormatter.format_array(content)
            self.elements.append(Preformatted(text, self.body_style))
        elif isinstance(content, torch.Tensor):
            text = tensorFormatter.TensorFormatter.format_tensor(content)
            self.elements.append(Preformatted(text, self.body_style))
        elif isinstance(content, plt.Figure):
            buf = io.BytesIO()
            content.savefig(buf, format='png')
            buf.seek(0)
            return Image(buf)
        elif isinstance(content, bytes):
            return Image(io.BytesIO(content))
        elif isinstance(content, PILImage.Image):
            buf = io.BytesIO()
            content.save(buf, format='PNG')
            buf.seek(0)
            return Image(buf)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def add_side_by_side(self, left_content, right_content):
        left_element = self._create_element(left_content)
        right_element = self._create_element(right_content)

        frame_width = self.doc.width / 2 - 6
        frame_height = self.doc.height

        left_frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, frame_width, frame_height, id='left')
        right_frame = Frame(self.doc.leftMargin + frame_width + 12, self.doc.bottomMargin, frame_width, frame_height, id='right')

        left_story = [left_element]
        right_story = [right_element]

        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(left_frame.addFromList(left_story, self.doc.canv))
        self.elements.append(right_frame.addFromList(right_story, self.doc.canv))
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
    report.add_to_report(np.ndarray((2,4,3,3)), "Numpy Array Heading")

    # Adding a matplotlib figure with a heading
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    report.add_to_report(fig, "Matplotlib Figure Heading")

    # Adding an image from path
    report.add_to_report("img.png", "Image from Path Heading")

    # Adding an image object using PIL
    pil_image = PILImage.new('RGB', (60, 30), color = (73, 109, 137))
    report.add_to_report(pil_image, "PIL Image Object Heading")

    # Adding two elements side by side
    report.add_side_by_side("Left Element", "Right Element")
    report.add_side_by_side(123, [4, 5, 6])
    report.add_side_by_side(torch.tensor([1, 2, 3]), np.array([4, 5, 6]))

    # Finalize the report
    report.build_report()
