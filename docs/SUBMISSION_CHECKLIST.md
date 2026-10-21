# ICCICA 2027 — submission checklist and fill-in sheet

Paper: "What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for
Diabetes from Population Health Surveys"

Everything below is either **done** (green) or an **author action** (orange). Nothing in the
orange list can be done by the agent: it needs your names, affiliations, funding line, and
your CMT account.

---

## 1. Fill in the author block — `manuscript/manuscript.tex` lines 17-21

```latex
\author{\IEEEauthorblockN{First Author, Second Author}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering} \\
\textit{Institution Name} \\
City, Country \\
{first.author@inst.edu}}}
```

- Corresponding author is marked by CMT (checkbox), not in the PDF.
- Use the e-mail that matches the CMT account for every author; CMT rejects mismatches.
- IEEE membership status is entered in CMT (it changes the registration fee), not in the PDF.
- The PDF itself must carry **no** identifying information beyond the author block.

## 2. Back matter — only one placeholder remains

The manuscript now carries the Acknowledgments (IEEE AI-use disclosure), the Data and Code
Availability statement, and the ethics sentence. The CRediT author-contribution statement,
the funding statement and the conflict-of-interest block were removed, so the **author block
is the only thing left to fill in**. If the venue later requests a CRediT statement, use:

- Author A: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing (original draft, review and editing)
- Author B: Data curation, Resources, Writing (review and editing)

## 3. Repository release (optional, not blocking)

The Data and Code Availability statement now reads: the code, run manifests, logged results
and the SHA-256 artifact manifest "are held in a private repository for review and will be
made public on acceptance". That is accurate today — `https://github.com/Aldrin7/brfss-diabetes-screening-iccica2027`
returns HTTP 404 (private or not yet created).

- For the **submission**, nothing to do: the statement is honest as written.
- For the **camera-ready (1 Jan 2027)**, make the repository public and change the sentence to
  "are released at", optionally adding a Zenodo DOI to \cite{repo} for a citable release.

The repository already contains: `src/`, `run_*.py`, `results/logs/*.json` (all 15 logged
run files), `assets/figures/`, `requirements.txt`, and the SHA-256 artifact manifest.
`data/raw/` is intentionally excluded (the BRFSS files are redistributed by CDC/UCI/Kaggle
under CC0 — see the README for the download instructions).


## 5. Build the two submission PDFs (already done, re-run after edits)

```powershell
python src/make_template_docx.py                                   # manuscript/manuscript.docx
python src/make_template_docx.py --anon --out manuscript/manuscript_anon.docx
python src/audit_numbers.py                 # 110 machine-checked numbers, 0 failures
python src/check_tex.py                     # citations, refs, labels, structure
```

Artifacts:

| File | Use |
|---|---|
| `manuscript/manuscript_anon.pdf` | **upload this** to CMT (double-blind) |
| `manuscript/manuscript.pdf` | camera-ready PDF (A4, two columns), keep the author block |
| `manuscript/manuscript.tex` | authoritative source of both |
| `manuscript_template.docx` | manuscript on the conference's own A4 Word template (`conference-template-a4.docx`) |
| `manuscript.docx` | Word version, single column — for co-author review and annotation |
| `manuscript_2col.docx` | Word version, two columns — matches the PDF layout, 6 pages |

### The conference A4 template

`conference-template-a4.docx` is the IEEE conference "live" template in A4 (Strict OOXML, so
it is namespace-converted before use). `python src/make_template_docx.py` rebuilds the
manuscript **on that template** — its styles, A4 page setup, margins, footer and numbering
are reused verbatim:

| Content | Template style |
|---|---|
| Title | `paper title` |
| Author block | `Author` (one line each) |
| Abstract / Keywords | `Abstract` / `Keywords` |
| Sections / subsections | `Heading 1` (auto Roman) / `Heading 2` (auto lettered) |
| Body | `Body Text` |
| Table label + caption | `table head` (auto "TABLE I.") |
| Table header / body cells | `table col head` / `table copy` |
| Figure caption | `figure caption` (auto "Fig. 1.") |
| Back matter | `Heading 5` |
| References | `references` (auto "[1]") |

Captions and references are **not** numbered manually — the template's own numbering
definitions supply "TABLE I.", "Fig. 1.", "[1]", "I." and "A.", so the numbers stay
consistent if content is added. Tables are real Word tables with IEEE rules, fixed column
widths, a repeating header row and merged group rows; the figure is embedded at 4.2 in.

The body is **two-column** (A4, 0.75 in margins, 3.28 in columns) with the title block
spanning the full width, which is the IEEE conference layout the CFP describes. Both the
named and the anonymized build come to **6 pages**. Tables are sized to the column width,
carry a repeating header row, are never split across a column break, and the group rows
span the table width; the figure is placed at 3.15 in inside the column.

If the organizers prefer the plain two-column IEEEtran (US Letter) rendering instead, that
build is kept in `archive/ieeetran_2col/` and can be regenerated with
`python src/compile_ieee.py --max 6`.

The Word versions are produced by `python src/make_docx.py [--twocol]`: IEEE-numbered
citations, all four tables and the figure as native Word objects, the 17 references as a
numbered list, and every number rendered as selectable text (inline LaTeX math is converted
to plain Unicode) so the document can be read, searched and tracked for changes. The PDF
remains the authoritative camera-ready artifact.

Current state: 6 pages each, US Letter 612x792 pt, 4 tables, 1 figure, 15 references, no
undefined citations or cross-references.

## 6. CMT submission steps — `cmt3.research.microsoft.com/ICCICA2027/Submission/Index`

1. Create/sign in to a CMT account for every author (each needs their own account and email).
2. Create a new submission; choose **Track 4 - Data Science, Machine Learning and Deep Learning**.
3. Title: copy verbatim from `manuscript/manuscript.tex` line 14.
4. Abstract: copy the text of the `abstract` environment (206 words, no citations inside).
   Keywords: `diabetes screening, machine learning, calibration, fairness auditing, SHAP, BRFSS, health equity, ethical AI` (8 terms; the CFP asks for up to 5 for the title page — trim to the first five if CMT rejects more).
5. Upload `manuscript_anon.pdf` as the only PDF. Do not upload the named version or a supplementary
   code archive in a way that reveals the author.
6. Enter every author's name, e-mail, affiliation and country **exactly** as in the PDF.
7. Declare conflicts of interest (none), and select the corresponding-author flag.
8. Choose the IEEE membership status per author — it determines the registration fee.
9. Add the topic keywords the track expects (machine learning, fairness, calibration, healthcare).
10. Submit before the extended deadline and keep the confirmation e-mail.

## 7. Pre-submission checks already verified

- [x] Page size US Letter, two columns, IEEEtran 10pt, margins per the CFP
- [x] <= 6 pages including figures, tables and references (no over-length charge)
- [x] All numbers in the manuscript machine-verified against the logged runs (121 checks: 69 + 41 + 11, 0 failures)
- [x] No undefined citations, no undefined references, no unused labels, no unused bibliography entries
- [x] Figures at >= 300 DPI (effective 578-737 DPI at placed size)
- [x] All interpretability plots contain only the 21 BRFSS features (verified for the round-6 review point)
- [x] Anonymized build contains no author name, affiliation, e-mail or repository URL
- [x] IEEE AI-use disclosure present in the Acknowledgments; AI tools not listed as authors
- [x] Ethics statement (public CC0 data, no IRB required) inside the Data and Code Availability section
- [x] No unresolved placeholders in the manuscript except the author block
- [x] Data and Code Availability restated to "held in a private repository for review and will be made public on acceptance" (matches the 404 on the public URL)
- [x] Eight review rounds completed; round 8 (run on the 6-page submission artifact) recommends acceptance after minor revisions, and all eight questions are answered in `paperreview_responses_round8.md` — the contradicting-label caveat, per-band AUC/ECE/net benefit, per-group calibration transport, Venn–Abers detail, SaML and multi-task citations are all in the paper


## 8. After acceptance (timeline)

| Date | Action |
|---|---|
| 15 Dec 2026 | acceptance notification |
| 1 Jan 2027 | camera-ready PDF (6-8 pages, IEEE copyright form signed) |
| 27 Jan 2027 | early registration (student INR 7,400 / regular INR 9,000) |
| 10 Feb 2027 | late registration |
| 11-12 Mar 2027 | conference, PIET Samalkha (hybrid); at least one author must register and present |

Prepare the presentation slides from the four figures already in `assets/figures/`:
`shap_top_features.png`, `calibration_curve.png`, `decision_curve.png`, `fairness_by_age.png`.
