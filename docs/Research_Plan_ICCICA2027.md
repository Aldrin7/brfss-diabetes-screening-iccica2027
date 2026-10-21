# Research Plan — ICCICA 2027 (Revised)

## Finalized Topic

**Title (working):** "An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys: Associations, Explainability, and Ethical Implications"

**Track:** Track 4 — Data Science, Machine Learning and Deep Learning (ICCICA 2027). Theme fit: "Computational Intelligence: Innovations, Applications, and Ethical Implications in a Rapidly Evolving Digital World" — healthcare screening carries direct ethical implications (health equity, proxy variables, screening vs. diagnosis).

**Venue:** 3rd International Conference on Computational Intelligence & Computing Applications (ICCICA-2027), PIET, Samalkha, Haryana, India; 11–12 March 2027; technically sponsored by IEEE Delhi Section; proceedings submitted to IEEE Xplore. Conference Record No. 73185. Double-blind review, min. 3 reviewers/paper, acceptance rate ~24.7% (2024/2025 editions).

## Part 3 — Project Inputs (filled)

### Venue
- Target venue: ICCICA 2027 (IEEE conference proceedings)
- Article type: Full conference paper
- Page/word limit: 6 pages (incl. figures, tables, references); up to 2 extra pages at charge
- Abstract: free-form single paragraph, IEEE norm (~150–250 words); no citations in abstract (verify in guide)
- Formatting: IEEE conference template, US Letter 8.5"x11", double column, single spaced, 10pt Times New Roman; margins 0.75" (1" top on title page); PDF via Microsoft CMT
- Citation style: IEEE numbered-sequential [1], [2], ...
- Reference-count norm: ~25–35 verified (6 pages)
- Figure resolution: ≥300 DPI (600 for line art), TIFF/EPS/PNG
- Thousands/decimal: Anglo-English (comma thousands, period decimal) — confirm in IEEE guide
- Required statements: data availability, code availability, competing interests, author contributions (CRediT), funding, ethics/IRB, AI-use disclosure
- Venue AI-disclosure policy: VERIFY by search before submission (IEEE authorship/AI-use policy)

### Research
- Domain: Computational intelligence / applied ML / AI ethics & fairness in healthcare screening
- Topic constraints: empirical, real public data, verifiable citations, reproducible; screening/diagnostic framing (per author style guide), NOT causal/forecasting/diagnostic
- Dataset: CDC Diabetes Health Indicators (BRFSS 2015), UCI ML Repository — https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators ; DOI 10.24432/C53919; cleaned from CDC BRFSS 2015 (441,455 raw respondents → 253,680 listwise-complete); 21 features; target Diabetes_binary (0 = no diabetes, 1 = prediabetes or diabetes) or Diabetes_012 (3 classes); no missing values in cleaned file; sensitive attributes available: Sex, Age (13-level), Income (8-level), Education (6-level); NOTE: race is NOT present in this cleaned subset — stated as a limitation and an optional extension using raw BRFSS XPT files. License: CC0 (public domain) for the cleaned Kaggle/UCI release; raw CDC BRFSS data is U.S. Government public domain. Provenance verified (CDC source, 2015 wave, listwise deletion, Alex Teboul cleaning protocol).
- Compute: CPU (local Windows), Python 3.x, scikit-learn, XGBoost, SHAP
- Time budget/deadline: full paper submission 12 Oct 2026
- Hard constraints: 6-page limit; no fabricated numbers/citations; multi-seed (≥3, prefer 5) mean±std; association-not-causation framing; screening-level only

### Team
- Author list/order: [TBD — fill before writing]
- Corresponding author: [TBD]
- Affiliations: [TBD]
- Data/code public: yes — intended for Zenodo (token on file) + GitHub (token on file); data already public (CDC/UCI/Kaggle)

## Research Questions
1. Driver identification: Which BRFSS health/lifestyle/demographic features most strongly associate with diabetes status, and how stable are these attributions across seeds?
2. Fairness: How do model performance and error rates differ across protected/stratifying groups (sex, age group, income, education), and which features drive disparities?
3. Screening utility: Does a simple, interpretable baseline (logistic regression) achieve competitive performance vs. a tree ensemble, supporting a short-questionnaire screening tool?
4. Ethical implications: What are the risks of deploying such a screening tool (health equity, proxy variables, false positives/negatives), and what safeguards (transparency, human oversight, use boundaries) are warranted?

## Models (scope declaration, pre-Gate C)
- M0: Logistic Regression (interpretable baseline, L2, scaled) — "humble baseline" move
- M1: Random Forest
- M2: Gradient Boosting (HistGradientBoosting or XGBoost)
- Ablations: feature groups (demographic-only; +clinical history; +lifestyle/behavior; full); with vs. without sensitive attributes (Sex/Age/Income/Education)
- Seeds: 5 (fixed seeds, deterministic); report mean±std; paired Wilcoxon across folds where applicable

## Evaluation Protocol
- Stratified 5-fold CV on training portion + a held-out stratified test set (20%); group-aware reporting by Sex, Age group, Income, Education
- Metrics: accuracy, F1, ROC-AUC, precision, recall; fairness: demographic parity difference, equalized odds difference, calibration by group
- Explainability: SHAP (TreeSHAP for trees; LinearExplainer for LR); feature importance; stability of top-k features across seeds
- Scenario: stylized perturbation of modifiable risk factors (e.g., +BMI, smoking, physical activity), labeled as sensitivity probes bounded by the historical feature range, not projections

## Deliverables / Folder Structure (Part 6/8)
project_root/
├── data/ (raw BRFSS CSV, processed)
├── notebooks/ (EDA)
├── src/ (preprocessing.py, features.py, models.py, train.py, evaluate.py, fairness.py, explain.py, utils.py)
├── configs/ (YAML)
├── checkpoints/
├── results/{figures,tables,logs}/
├── future_scope/
├── requirements.txt (pinned)
├── README.md
└── run_pipeline.py

## Verification Gates (Part 4)
Gates A–G in order, stop + Verification Report at each, await acknowledgment. Gate A data provenance first.

## Limitations (to be stated up front, Rule 3)
- Single survey wave (BRFSS 2015); no temporal/external validation; multi-year extension (2017–2023) listed as future work
- Self-reported survey data; no clinical biomarkers (HbA1c, glucose) in this subset
- Race/ethnicity NOT present in this cleaned subset — fairness analysis limited to sex, age, income, education; race-level analysis requires raw BRFSS XPT (optional extension, scope TBD)
- Association, not causation; no causal claims
- Single train/test split within CV; acknowledge
- Class imbalance (diabetes prevalence ~13.9%); handle via stratification + class weights; report
- Screening-level tool only; not validated for individual clinical diagnosis; complements, not replaces, clinical assessment

## Author Style
Append Author_Writing_Style_Guide.txt as Part 8B (colon-structured title, 13-step abstract, numbered sections 1.1–5, hedged verbs, "We acknowledge/We caution", bold lead-ins, (i)-(iv) contributions, complement-not-replace close, back matter in MDPI order). Integrity rules (Part 2) always take precedence.

## Next Steps
1. Fill author/affiliation fields (Part 3 Team).
2. Obtain & read IEEE author guidelines + ICCICA 2027 CFP details; verify AI-disclosure policy.
3. Gate A: acquire BRFSS 2015 cleaned CSV, verify schema/counts/class balance/prevalence, document provenance.
4. Proceed through Gates B–G.
