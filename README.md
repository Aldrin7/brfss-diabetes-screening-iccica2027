# Diabetes Screening from BRFSS 2015 — ICCICA 2027

Reproducible pipeline for the manuscript:
"An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys".

## Data
CDC Diabetes Health Indicators (BRFSS 2015), UCI ML Repository, DOI 10.24432/C53919.
Download via Kaggle: `alexteboul/diabetes-health-indicators-dataset` -> place
`diabetes_binary_health_indicators_BRFSS2015.csv` in `data/raw/`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_pipeline.py          # 5 seeds x 3 models + ablation (~4 min)
python run_novel.py             # temporal validation, minimal questionnaire, threshold opt (~1.5 min)
python run_audit.py             # leakage sensitivity, calibration, fairness w/ inference (~10 min)
python run_audit2.py            # hyperparameter search, wave drift, feature selectors (~20 min)
python run_audit3.py            # post-hoc recalibration, age-gap robustness (~15 s)
python run_audit4.py            # paired AUC bootstrap, scenario probes (~4 min)
python src/explain.py           # SHAP + figure
python src/gateA_verify.py      # Gate A data verification
python src/make_figures.py      # confusion matrix + fairness aggregate
python src/make_audit_figures.py  # calibration / leakage / fairness / drift figures
python src/make_pdf.py          # render manuscript.pdf for external review
```

## Outputs
- results/logs/results.json — per-seed metrics, fairness, ablation
- results/logs/novel_results.json — temporal validation, minimal questionnaire, thresholds
- results/logs/audit_results*.json — leakage, calibration, fairness inference, HP search, drift, selectors, scenarios
- results/logs/shap.json, fairness_agg.json
- results/figures/*.png
- manuscript.tex — IEEE conference manuscript (authoritative source)
- manuscript.pdf — rendered review copy (single-column; not the IEEE two-column camera-ready)

## Seeds
42, 7, 123, 2024, 99 (reported as mean +/- std).

## Integrity
All numbers trace to logged runs (results/logs/results.json). No fabricated metrics.
