from fpdf import FPDF
import tempfile
import os

class StrategyPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(220, 50, 50) # Rouge F1
        self.cell(0, 10, 'F1 PIT WALL OS - PRE-RACE STRATEGY BRIEF', 0, 1, 'C')
        self.set_draw_color(220, 50, 50)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(circuit_name, best_strat, total_laps, pit_loss, time_str, fig_pace=None):
    pdf = StrategyPDF()
    pdf.add_page()
    
    # Informations du Circuit
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f'GRAND PRIX : {circuit_name.upper()}', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f'Distance de course : {total_laps} tours', 0, 1)
    pdf.cell(0, 8, f'Perte aux stands moyenne : {pit_loss} secondes', 0, 1)
    pdf.ln(5)
    
    # Stratégie Optimale
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 102, 204) # Bleu
    pdf.cell(0, 10, 'STRATEGIE OPTIMALE RECOMMANDEE (Dynamic Programming)', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    compounds_str = " -> ".join(best_strat['compounds'])
    pits_str = ", ".join(map(str, best_strat['pit_laps'])) if best_strat['pit_laps'] else "Aucun"
    
    pdf.cell(0, 8, f'Nombre d\'arrets : {best_strat["stops"]}', 0, 1)
    pdf.cell(0, 8, f'Sequence de gommes : {compounds_str}', 0, 1)
    pdf.cell(0, 8, f'Fenêtres d\'arrets (Tours) : {pits_str}', 0, 1)
    pdf.cell(0, 8, f'Temps de course estime : {time_str}', 0, 1)
    pdf.ln(10)
    
    # Insertion du Graphique si existant
    if fig_pace:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'PROJECTION DU RYTHME (PACE)', 0, 1)
        # Sauvegarde temporaire de l'image (sans fond noir !)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig_pace.savefig(tmpfile.name, format="png", bbox_inches="tight")
            pdf.image(tmpfile.name, x=10, w=190)
        os.remove(tmpfile.name)
        
    # Export sécurisé en bytes pour Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp_pdf.name)
    
    return pdf_bytes