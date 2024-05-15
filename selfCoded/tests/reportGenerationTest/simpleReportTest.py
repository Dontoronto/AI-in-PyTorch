from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

def create_matrix_image(filename):
    # Create an image with white background
    img = PILImage.new('RGB', (100, 100), color = 'white')
    d = ImageDraw.Draw(img)

    # Load a font
    font = ImageFont.load_default()

    # Draw the matrix with highlighted parts
    d.rectangle([20, 20, 80, 80], outline="black", width=2)
    d.rectangle([40, 20, 60, 40], fill="yellow")
    d.rectangle([40, 40, 60, 60], fill="yellow")
    d.rectangle([20, 40, 40, 60], fill="yellow")
    d.rectangle([60, 40, 80, 60], fill="yellow")
    d.text((30, 10), "0", fill="black", font=font)
    d.text((50, 10), "1", fill="black", font=font)
    d.text((70, 10), "0", fill="black", font=font)
    d.text((30, 50), "1", fill="black", font=font)
    d.text((50, 50), "1", fill="black", font=font)
    d.text((70, 50), "1", fill="black", font=font)
    d.text((30, 90), "0", fill="black", font=font)
    d.text((50, 90), "0", fill="black", font=font)
    d.text((70, 90), "0", fill="black", font=font)

    # Save the image
    img.save(filename)

def create_pdf(pdf_file):
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    elements = []

    # Add some text
    styles = getSampleStyleSheet()
    text = "Hello, this is a report generated using ReportLab!"
    paragraph = Paragraph(text, styles['Normal'])
    elements.append(paragraph)

    # Add some space
    elements.append(Spacer(1, 12))

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))
    fig.savefig("plot.png")
    plt.close(fig)

    # Add the matplotlib figure to the PDF
    img = Image("plot.png")
    img.drawHeight = 4 * inch
    img.drawWidth = 6 * inch
    elements.append(img)

    # Add some space
    elements.append(Spacer(1, 12))

    # Add a table
    data = [['A', 'B', 'C'],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(table)

    # Add some space
    elements.append(Spacer(1, 12))

    # Create and add the custom matrix image
    matrix_image_file = "matrix.png"
    create_matrix_image(matrix_image_file)
    matrix_img = Image(matrix_image_file)
    matrix_img.drawHeight = 2 * inch
    matrix_img.drawWidth = 2 * inch
    elements.append(matrix_img)

    # Build the PDF
    doc.build(elements)

# Generate the PDF
create_pdf('report.pdf')
