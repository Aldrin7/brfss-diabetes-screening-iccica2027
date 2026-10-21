# Diabetes Screening from BRFSS 2015 — ICCICA 2027

Reproducible pipeline for the manuscript:
"An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys: Leakage, Calibration, and Fairness Audits".

## Data
CDC Diabetes Health Indicators (BRFSS 2015), UCI ML Repository, DOI 10.24432/C53919.
Download via Kaggle: `alexteboul/diabetes-health-indicators-dataset` -> place
`diabetes_binary_health_indicators_BRFSS2015.csv` in `data/raw/`.
Out-of-time waves (2017/2019/2021) come from Kaggle `vanthieu/diabetes-health-indicators-brfss-2015-2023`
-> place the unzipped files in `data/raw/brfss_multi/`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_pipeline.py             # 5 seeds x 3 models + ablation (~4 min)
python run_novel.py                # temporal validation, minimal questionnaire, threshold opt (~1.5 min)
python run_audit.py                # leakage sensitivity, calibration, fairness w/ inference (~10 min)
python run_audit2.py               # hyperparameter search, wave drift, feature selectors (~20 min)
python run_audit3.py               # post-hoc recalibration, age-gap robustness (~15 s)
python run_audit4.py               # paired AUC bootstrap, scenario probes (~4 min)
python run_audit5.py               # leakage protocols, 4 calibrators, AUPRC, decision curve (~12 min)
python run_audit5b.py              # SHAP stability, design effect, severity strata (~2 min)
python run_audit6.py               # per-wave calibration transport + subgroup drift (~3 min)
python run_audit7.py               # group-specific thresholds, scenario subgroups, manifest (~1 min)
python run_audit8.py               # constrained threshold selection, severity stratification (~25 s)
python run_audit9.py               # transport vs refit calibration per wave (~40 s)
python src/explain.py              # SHAP + figure
python src/gateA_verify.py         # Gate A data verification
python src/make_figures.py         # confusion matrix + fairness aggregate
python src/make_audit_figures.py   # calibration / leakage / fairness / drift / selector figures
python src/make_dca_figure.py      # decision-curve net benefit figure
python src/compile_ieee.py         # REAL IEEEtran build + page-count check (authoritative PDF)
python src/compile_ieee.py --anon   # also build the double-blind anonymized PDF
python src/make_anon.py            # anonymized double-blind PDF only
python src/make_docx.py            # Word version (single column, editable)
python src/make_docx.py --twocol   # Word version, two columns like the PDF
python src/make_template_docx.py   # Word version on conference-template-a4.docx
python src/check_tex.py            # LaTeX structure / citation / cross-reference check
python src/audit_numbers.py        # machine-check every number in the manuscript against run logs (121 checks)
python src/audit_round8.py         # round-8 review computations: per-band AUC/ECE/net benefit, calibration transport
```

## Camera-ready PDF (authoritative)
`manuscript.pdf` and `manuscript_anon.pdf` at the repository root are **true IEEEtran
builds** (US Letter, two-column, 10pt), not the legacy ReportLab review mock
(`src/make_pdf.py`, still kept for quick visual diffs). Regenerate both with:

```bash
python src/compile_ieee.py --anon --max 6
```

`compile_ieee.py` exits non-zero if the paper exceeds the 6-page ICCICA 2027 limit
(including figures, tables and references) and prints over/underfull box warnings.
`src/make_anon.py` strips the author block, the repository URL and the author name from
the bibliography, and verifies that no identifier remains in the output.


## Outputs
- results/logs/results.json — per-seed metrics, fairness, ablation
- results/logs/novel_results.json — temporal validation, minimal questionnaire, thresholds
- results/logs/audit_results*.json — leakage protocols, calibrators, fairness inference,
  HP search, wave drift, selectors, scenarios, constrained thresholds, manifest
- results/logs/shap.json, fairness_agg.json, gateA_report.json
- results/logs/audit_results13.json, age_band_supports.csv — round-8 per-band AUC/ECE/net
  benefit, per-group vs global calibration transport, age-band definitions and supports
- results/figures/*.png, figures/*.png
- manuscript.tex — IEEE conference manuscript (authoritative source)
- manuscript.pdf — IEEEtran build, 6 pages, for camera-ready use
- manuscript_anon.pdf — anonymized double-blind build, 6 pages, for the CMT submission
- manuscript_template.docx — manuscript on the conference's own A4 Word template
  (conference-template-a4.docx), 7 pages
- manuscript.docx — Word version, single column, editable (7 pages)
- manuscript_2col.docx — Word version, two columns matching the PDF layout (6 pages)
- paperreview_review_round*.md, paperreview_responses_round*.md — eight external review rounds
  and the point-by-point response records

## Submission target
ICCICA 2027 (3rd International Conference on Computational Intelligence & Computing
Applications), 11-12 March 2027, PIET Samalkha, Haryana, India; technically sponsored by
IEEE Delhi Section; IEEE Xplore, Conference Record No. 73185.

| Item | Requirement | State |
|---|---|---|
| Format | US Letter, IEEEtran conference, 2-column, 10pt, 0.75in margins (1in top on title page) | met (612x792 pt, verified) |
| Length | <= 6 pages incl. figures, tables, references | met (6 pages, both builds) |
| Extra pages | up to 2 allowed at INR 1,000/page | not needed |
| Review | double-blind, min. 3 reviewers, IEEE plagiarism check | anonymized build provided |
| Submission | Microsoft CMT: `cmt3.research.microsoft.com/ICCICA2027/Submission/Index` | author action |
| Track | Track 4 - Data Science, Machine Learning and Deep Learning | selected |
| Dates | submission 12 Oct 2026 (extension granted), notification 15 Dec 2026, camera-ready 1 Jan 2027, early registration 27 Jan 2027, late 10 Feb 2027 | from the official CFP |
| Registration | student INR 7,400/9,000; regular INR 9,000/11,000; co-author INR 1,500; one author must register | author action |


## Seeds
42, 7, 123, 2024, 99 (reported as mean +/- std).

## Integrity
Every number in the manuscript is machine-checked against the logged runs by
`src/audit_numbers.py` (121 checks: 69 + 41 + 11). No fabricated metrics. The SHA-256 manifest of
all artifacts is regenerated by `run_audit7.py`.

## Known data constraints (documented in the paper)
- The cleaned release carries no survey design variables (weights, strata, PSUs), so
  population-level prevalence claims are not supported; design-effect sensitivity is
  reported instead.
- Race/ethnicity is absent from the cleaned release, so fairness analysis covers
  sex, age, income, and education only.
- The target collapses prediabetes and diabetes; severity-stratified analysis is used
  as a proxy for a condition-level breakdown.
- The 2023 wave lacks four features and is excluded from out-of-time validation.

