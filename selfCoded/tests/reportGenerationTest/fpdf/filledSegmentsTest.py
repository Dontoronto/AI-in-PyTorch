from fpdf import FPDF

import lorem # Use this package to showcase long texts
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', '', 16)
pdf.cell(w=0, h=50, txt="This and the below cells are regular cells." , border=1, ln=1)
pdf.cell(w=0, h=50, txt="Example: " + lorem.text(), border=1, ln=1)
pdf.multi_cell(w=0, h=50, txt="This and the below cells are multi cells.", border=1, )
pdf.multi_cell(w=0, h=5, txt="Example: " + lorem.text(), border=1, )
pdf.output(f'./example.pdf', 'F')