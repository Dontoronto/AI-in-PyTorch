from fpdf import FPDF

# Custom class to overwrite the header and footer methods
class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, 'Header', 1, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, 'Footer', 1, 0, 'C')
pdf = PDF() # Instance of custom class
pdf.add_page()
pdf.set_font('Arial', '', 12)
pdf.cell(w=0, h=255, txt = "Body", border = 1, ln = 1, align = 'C')
pdf.output(f'./example.pdf', 'F')