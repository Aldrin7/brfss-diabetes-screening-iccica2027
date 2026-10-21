import re, os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, KeepTogether, FrameBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "manuscript.tex")
OUT = os.path.join(ROOT, "manuscript.pdf")
FIGS = os.path.join(ROOT, "results", "figures")
FIGDIR = os.path.join(ROOT, "figures")

raw = open(TEX, encoding="utf-8").read()
raw = re.sub(r"%.*", "", raw)

REFMAP = {}

def strip_inline(s):
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(REFMAP.get(m.group(1), "?")), s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    for cmd in ["textit", "textbf", "emph", "textrm", "textsc", "mathrm", "texttt"]:
        s = re.sub(r"\\" + cmd + r"\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("$", "")
    s = s.replace("\\,", " ").replace("\\%", "%").replace("{", "").replace("}", "")
    s = re.sub(r"~", " ", s)
    return re.sub(r"\s+", " ", s).strip()

m = re.search(r"\\title\{([^}]*)\}", raw)
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

PW, PH = letter
LM = RM = 0.55 * inch
TM = 0.5 * inch
BM = 0.55 * inch
GUT = 0.22 * inch
COLW = (PW - LM - RM - GUT) / 2.0

styles = getSampleStyleSheet()
S = {}
S["title"] = ParagraphStyle("title", parent=styles["Title"], fontName="Times-Bold",
                             fontSize=13.5, leading=15.5, spaceAfter=8, alignment=TA_CENTER)
S["h1"] = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=9.5,
                         leading=11, spaceBefore=7, spaceAfter=3, alignment=TA_LEFT)
S["h2"] = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Times-Italic", fontSize=8.5,
                         leading=10, spaceBefore=5, spaceAfter=2, alignment=TA_LEFT)
S["body"] = ParagraphStyle("body", parent=styles["BodyText"], fontName="Times-Roman",
                           fontSize=8.2, leading=9.6, spaceAfter=3, alignment=TA_JUSTIFY)
S["abs"] = ParagraphStyle("abs", parent=styles["BodyText"], fontName="Times-Roman",
                          fontSize=8.2, leading=9.6, spaceAfter=3, alignment=TA_JUSTIFY)
S["cap"] = ParagraphStyle("cap", parent=styles["BodyText"], fontName="Times-Italic",
                          fontSize=7.2, leading=8.4, spaceAfter=3, alignment=TA_CENTER)
S["tab"] = ParagraphStyle("tab", parent=styles["BodyText"], fontName="Times-Roman",
                          fontSize=7.0, leading=8.2, spaceAfter=0, alignment=TA_LEFT)

story = []
story.append(Paragraph(title, S["title"]))
story.append(Paragraph("<b>Abstract</b>—&nbsp;" + abstract, S["abs"]))
story.append(Paragraph("<b>Index Terms</b>—&nbsp;" + keywords, S["abs"]))
story.append(Spacer(1, 4))
story.append(FrameBreak())

FIG_MAP = {
    "shap_top_features.png": None,
    "calibration_curve.png": None,
    "age_gap_support.png": None,
    "fairness_by_age.png": None,
    "leakage_sensitivity.png": None,
    "feature_selectors.png": None,
    "wave_drift.png": None,
}

def make_table(tabdata, caption):
    ncols = max(len(r) for r in tabdata)
    for r in tabdata:
        while len(r) < ncols:
            r.append("")
    cell = [[Paragraph(c, S["tab"]) for c in row] for row in tabdata]
    t = Table(cell, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [Paragraph("<b>TABLE</b>&nbsp; " + caption, S["cap"]), t, Spacer(1, 4)]

token_re = re.compile(r"(\\section\*?\{[^}]*\}|\\subsection\*?\{[^}]*\}|\\begin\{table\}.*?\\end\{table\}"
                      r"|\\begin\{thebibliography\}.*?\\end\{thebibliography\}"
                      r"|\\begin\{figure\}.*?\\end\{figure\})", re.S)

_parts = token_re.split(body)
_tn = 0
_fn = 0
for _p in _parts:
    if _p.strip().startswith("\\begin{table}"):
        _tn += 1
        _lm = re.search(r"\\label\{([^}]*)\}", _p)
        if _lm:
            REFMAP[_lm.group(1)] = _tn
    elif _p.strip().startswith("\\begin{figure}"):
        _fn += 1
        _lm = re.search(r"\\label\{([^}]*)\}", _p)
        if _lm:
            REFMAP[_lm.group(1)] = _fn

for part in _parts:
    part = part.strip()
    if not part:
        continue
    if part.startswith("\\section"):
        mm = re.search(r"\\section\*?\{(.*)\}", part, re.S)
        if mm:
            story.append(Paragraph(strip_inline(mm.group(1)), S["h1"]))
    elif part.startswith("\\subsection"):
        mm = re.search(r"\\subsection\*?\{(.*)\}", part, re.S)
        if mm:
            story.append(Paragraph(strip_inline(mm.group(1)), S["h2"]))
    elif part.startswith("\\begin{table}"):
        tm = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", part, re.S)
        cm = re.search(r"\\caption\{(.*?)\}\s*\\label", part, re.S)
        if tm:
            rows = []
            for row in tm.group(1).strip().split(r"\\"):
                row = row.strip()
                if not row:
                    continue
                rows.append([strip_inline(c.strip()) for c in row.split("&")])
            if rows:
                story += make_table(rows, strip_inline(cm.group(1)) if cm else "")
    elif part.startswith("\\begin{thebibliography}"):
        story.append(Paragraph("References", S["h1"]))
        refno = 0
        for bm in re.finditer(r"\\bibitem\{[^}]*\}\s*(.*?)(?=\\bibitem|\Z)", part, re.S):
            ref = strip_inline(bm.group(1))
            if ref:
                refno += 1
                story.append(Paragraph("[%d]&nbsp;%s" % (refno, ref), S["body"]))
    elif part.startswith("\\begin{figure}"):
        im = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", part)
        cap = re.search(r"\\caption\{(.*?)\}\s*\\label", part, re.S)
        fname = im.group(1) if im else None
        if fname:
            for base in (FIGDIR, FIGS):
                p = os.path.join(base, fname)
                if os.path.exists(p):
                    try:
                        imobj = Image(p)
                        w = COLW
                        imobj.drawWidth = w
                        imobj.drawHeight = imobj.imageHeight * w / imobj.imageWidth
                        maxh = 150
                        if imobj.drawHeight > maxh:
                            imobj.drawHeight = maxh
                            imobj.drawWidth = imobj.imageWidth * maxh / imobj.drawHeight
                        story.append(imobj)
                    except Exception:
                        pass
                    break
        if cap:
            story.append(Paragraph("<b>Fig.</b>&nbsp;" + strip_inline(cap.group(1)), S["cap"]))
        story.append(Spacer(1, 3))
    else:
        txt = strip_inline(part)
        if txt and len(txt) > 20:
            story.append(Paragraph(txt, S["body"]))

doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=LM, rightMargin=RM,
                      topMargin=TM, bottomMargin=BM,
                      title=title, author="ICCICA 2027 submission draft")
frame1 = Frame(LM, BM, COLW, PH - TM - BM, id="col1",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
frame2 = Frame(LM + COLW + GUT, BM, COLW, PH - TM - BM, id="col2",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="twocol", frames=[frame1, frame2])])
doc.build(story)
print("Built", OUT)