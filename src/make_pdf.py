import re, os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "manuscript.tex")
OUT = os.path.join(ROOT, "manuscript.pdf")
FIGS = os.path.join(ROOT, "results", "figures")

raw = open(TEX, encoding="utf-8").read()
raw = re.sub(r"%.*", "", raw)

def strip_inline(s):
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\ref\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    for cmd in ["textit", "textbf", "emph", "textrm", "textsc", "mathrm", "texttt", "texttt"]:
        s = re.sub(r"\\" + cmd + r"\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

m = re.search(r"\\title\{(.*)\}", raw, re.S)
title = strip_inline(m.group(1)) if m else "Untitled"
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", raw, re.S)
abstract = strip_inline(m.group(1)) if m else ""
m = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", raw, re.S)
keywords = strip_inline(m.group(1)) if m else ""

body = raw
for pat in [r"\\title\{.*?\}", r"\\begin\{abstract\}.*?\\end\{abstract\}",
            r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", r"\\maketitle",
            r"\\author\{.*?\}", r"\\IEEEauthorblockN\{.*?\}", r"\\IEEEauthorblockA\{.*?\}"]:
    body = re.sub(pat, "", body, flags=re.S)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=15, spaceAfter=10, alignment=1)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=12, spaceBefore=12, spaceAfter=5)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10.5, spaceBefore=9, spaceAfter=3)
body_style = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.5, leading=11.5, spaceAfter=4, alignment=4)
abs_style = ParagraphStyle("A", parent=styles["BodyText"], fontSize=9.5, leading=11.5, spaceAfter=4,
                            alignment=4, leftIndent=8, rightIndent=8)

story = [Paragraph(title, title_style), Spacer(1, 5),
         Paragraph("<b>Abstract—</b>" + abstract, abs_style), Spacer(1, 3),
         Paragraph("<b>Index Terms—</b>" + keywords, abs_style), Spacer(1, 8)]

FIG_MAP = {
    "shap_top_features.png": "SHAP top-12 features",
    "calibration_curve.png": "Reliability diagrams before and after recalibration",
    "fairness_by_age.png": "TPR and FPR by age band",
    "age_gap_support.png": "Age TPR gap under minimum-support constraint",
    "leakage_sensitivity.png": "Leakage sensitivity (grouped vs stratified CV)",
    "feature_selectors.png": "Feature selector comparison",
    "wave_drift.png": "Out-of-time drift",
}

token_re = re.compile(r"(\\section\*?\{[^}]*\}|\\subsection\*?\{[^}]*\}|\\begin\{table\}.*?\\end\{table\}"
                      r"|\\begin\{thebibliography\}.*?\\end\{thebibliography\}"
                      r"|\\begin\{figure\}.*?\\end\{figure\})", re.S)
for part in token_re.split(body):
    part = part.strip()
    if not part:
        continue
    if part.startswith("\\section"):
        mm = re.search(r"\\section\*?\{(.*)\}", part, re.S)
        if mm:
            story.append(Paragraph(strip_inline(mm.group(1)), h1))
    elif part.startswith("\\subsection"):
        mm = re.search(r"\\subsection\*?\{(.*)\}", part, re.S)
        if mm:
            story.append(Paragraph(strip_inline(mm.group(1)), h2))
    elif part.startswith("\\begin{table}"):
        tm = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", part, re.S)
        if tm:
            table_data = []
            for row in tm.group(1).strip().split(r"\\"):
                row = row.strip()
                if not row:
                    continue
                table_data.append([strip_inline(c.strip()) for c in row.split("&")])
            if table_data:
                ncols = max(len(r) for r in table_data)
                for r in table_data:
                    while len(r) < ncols:
                        r.append("")
                t = Table(table_data, hAlign="CENTER", repeatRows=1)
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]))
                story.append(t)
                story.append(Spacer(1, 5))
    elif part.startswith("\\begin{thebibliography}"):
        story.append(Paragraph("References", h1))
        for bm in re.finditer(r"\\bibitem\{[^}]*\}\s*(.*?)(?=\\bibitem|\Z)", part, re.S):
            ref = strip_inline(bm.group(1))
            if ref:
                story.append(Paragraph(ref, body_style))
    elif part.startswith("\\begin{figure}"):
        im = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", part)
        cap = re.search(r"\\caption\{(.*?)\}\s*\\label", part, re.S)
        fname = im.group(1) if im else None
        if fname:
            for base in (FIGS, os.path.join(FIGS, ".."), os.path.join(ROOT, "figures")):
                p = os.path.join(base, fname)
                if os.path.exists(p):
                    try:
                        imobj = Image(p)
                        imobj.drawHeight = imobj.imageHeight * 210.0 / imobj.imageWidth
                        imobj.drawWidth = 210
                        if imobj.drawHeight > 230:
                            imobj.drawHeight = 230
                            imobj.drawWidth = imobj.imageWidth * 230 / imobj.drawHeight
                        story.append(imobj)
                    except Exception:
                        pass
                    break
        if cap:
            story.append(Paragraph("<i>" + strip_inline(cap.group(1)) + "</i>",
                                    ParagraphStyle("Cap", parent=body_style, fontSize=8, alignment=1)))
        story.append(Spacer(1, 4))
    else:
        txt = strip_inline(part)
        if txt and len(txt) > 20:
            story.append(Paragraph(txt, body_style))

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.6 * inch)
doc.build(story)
print("Built", OUT)