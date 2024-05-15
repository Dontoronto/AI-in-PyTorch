from fpdf import FPDF

# Margin
m = 10
MARGIN = m
# Page width: Width of A4 is 210mm
pw = 210 - 2*MARGIN
# Cell height
ch = 50
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)
pdf.cell(w=0, h=ch, txt="Cell 1", border=1, ln=1)
pdf.cell(w=(pw/2), h=ch, txt="Cell 2a", border=1, ln=0)
pdf.cell(w=(pw/2), h=ch, txt="Cell 2b", border=1, ln=1)
pdf.cell(w=(pw/3), h=ch, txt="Cell 3a", border=1, ln=0)
pdf.cell(w=(pw/3), h=ch, txt="Cell 3b", border=1, ln=0)
pdf.cell(w=(pw/3), h=ch, txt="Cell 3c", border=1, ln=1)
pdf.cell(w=(pw/3), h=ch, txt="Cell 4a", border=1, ln=0)
pdf.cell(w=(pw/3)*2, h=ch, txt="Cell 4b", border=1, ln=1)
pdf.set_xy(x=10, y= 220) # or use pdf.ln(50)
pdf.cell(w=0, h=ch, txt="Cell 5", border=1, ln=1)
pdf.output(f'./example.pdf', 'F')