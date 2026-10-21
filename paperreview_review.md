# Review: An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys

Venue: ICCICA

## Summary

The paper presents an interpretable, fairness-aware machine learning framework for diabetes screening using the publicly available 2015 BRFSS (n=253,680; 21 survey features). It benchmarks logistic regression, random forest, and histogram-based gradient boosting across five seeds, reports discrimination and subgroup fairness metrics, provides SHAP-based explanations and feature-group ablations, and adds three extensions: out-of-time validation on 2017–2021 waves, a minimal-questionnaire experiment using SHAP-ranked features, and threshold selection tailored to screening sensitivity. Overall AUCs cluster around 0.82–0.83, with tree ensembles modestly outperforming logistic regression and SHAP attributing most signal to general health, blood pressure, age, BMI, and cholesterol.

## Strengths

- Technical novelty and innovation
  - The paper integrates a practical, interpretable screening pipeline with SHAP explanations and fairness reporting, and introduces two useful extensions: out-of-time validation on later BRFSS waves and a compact “minimal questionnaire” analysis suggesting near-retention of AUC with 5–11 features.
  - Threshold selection aligned with screening priorities (target sensitivity) is a pragmatic addition for real-world deployment scenarios.
- Experimental rigor and validation
  - Large-scale dataset (n≈254k) with multi-seed re-splitting; consistent reporting of mean ± SD across five seeds for the main comparisons.
  - Sensible ablations by feature groups highlight where the predictive signal resides (clinical history dominates).
  - Out-of-time validation across multiple survey years strengthens claims of temporal stability.
- Clarity of presentation
  - Clear articulation of the intended use as screening (not diagnosis), limitations of cross-sectional survey data, and ethical boundaries.
  - SHAP results align with established risk factors, aiding interpretability and face validity.
- Significance of contributions
  - Diabetes screening at population scale is an important public health problem; the framework emphasizes reproducibility and interpretability on open data, potentially supporting public health outreach and hypothesis generation.
  - The minimal-questionnaire result has operational relevance for low-resource settings.

## Weaknesses

- Technical limitations or concerns
  - Survey design and weights are not incorporated despite BRFSS being a complex survey, limiting claims about population-level validity and subgroup fairness.
  - Duplicate feature profiles (≈9.5%) were not deduplicated or clustered for split assignment, risking leakage (train and test may share identical covariate vectors) and optimistic performance.
  - Fairness characterization is thin: very large TPR gaps (especially across 13 age groups) are reported but attributed to base-rate differences; however, TPR is, in principle, prevalence-invariant, calling for deeper diagnostic analysis (e.g., equalized odds, per-group calibration, uncertainty, power analyses).
  - Calibration is claimed in the abstract, but calibration metrics (Brier score, ECE/MCE, reliability curves) and groupwise calibration analyses are not reported.
- Experimental gaps or methodological issues
  - Minimal hyperparameter tuning may understate potential performance, and threshold selection results contain inconsistencies (e.g., sensitivity at the default threshold conflicts with the confusion matrix).
  - The minimal-questionnaire feature selection appears to rely on SHAP rankings computed on the test sample, raising potential selection bias; rankings should be derived from training data or through nested CV.
  - No head-to-head comparisons with closely related, recent non-laboratory screening work (e.g., NHANES-based models) or with parsimonious feature-selection filters beyond ablations; limited statistical testing (e.g., DeLong tests) across model AUCs.
- Clarity or presentation issues
  - Inconsistencies: limitations section still states “no temporal or external validation,” yet out-of-time validation is presented as a novel contribution; the threshold table’s “Default” row shows sensitivity 0.500 vs. 0.797 reported elsewhere.
  - Formatting artifacts and duplicated abstract snippets distract and obscure some details (e.g., incomplete equations, leftover LaTeX).
- Missing related work or comparisons
  - Limited discussion of survey-aware modeling and uncertainty (e.g., survey-weighted training and conformal UQ on complex surveys).
  - Fairness auditing best practices (multi-metric disparity with inference, minimum detectable effects, per-group calibration, and mitigation) are not integrated despite existing frameworks.
  - Recent BRFSS/NHANES screening studies and parsimonious feature-selection methods with comparable performance are not fully contextualized.

## Detailed Comments

- Technical soundness evaluation
  - The core supervised modeling is sound at a high level, and the reported AUCs and SHAP attributions are plausible and consistent with prior literature.
  - However, ignoring BRFSS survey weights undermines the representativeness of estimates (e.g., overall AUC, subgroup metrics). This is critical for population screening and fairness assessments.
  - Retaining exact duplicate profiles without ensuring fold-level isolation (e.g., grouping identical profiles during splitting) can leak information and inflate performance. A sensitivity analysis removing or grouping duplicates is warranted.
  - The fairness analysis relies mainly on TPR and accuracy gaps; extremely large TPR gaps (age up to ~0.82) suggest either severe model heterogeneity or instability due to low positive counts in some strata. Provide per-group supports and uncertainty (bootstrap CIs or conditional permutation tests) and consider equalized odds difference with inference. The assertion that TPR gaps “track base rates” is not technically well supported because recall is prevalence-invariant; more careful diagnostics (e.g., residual disagreement, proxy-variable mechanisms, groupwise calibration) are necessary.
  - Calibration (global and subgroup) is essential before threshold optimization for screening yet is not reported despite being listed in the abstract; add Brier score, reliability curves, ECE/MCE (overall and per group), and calibration-sensitive decision analyses.
- Experimental evaluation assessment
  - The multi-seed protocol is a plus; still, fixed, light hyperparameters and absence of tuned baselines (e.g., XGBoost/LightGBM with randomized search) may bias comparisons in favor of simplicity. A small, well-controlled hyperparameter search would strengthen conclusions regarding the competitiveness of logistic regression vs. ensembles.
  - Out-of-time validation across 2017–2021 is valuable; please clarify whether variable codings were fully harmonized, whether re-calibration by wave was attempted, and whether threshold selection was re-optimized per wave to maintain sensitivity targets.
  - The minimal-questionnaire experiment is operationally relevant, but to avoid look-ahead bias, the SHAP ranking should be computed on training data only (or via nested cross-validation), and results reported across seeds with CIs.
  - The threshold table includes a likely error (default threshold sensitivity reported as 0.500, conflicting with confusion-matrix recall ≈0.797 and earlier narrative that sensitivity falls from 0.797→0.901 when lowering the threshold); this needs correction.
- Comparison with related work (using the summaries provided)
  - NHANES non-laboratory screening (2606.16056) achieves AUC ≈0.809–0.820 and emphasizes calibration and decision-curve analysis; your AUCs are similar, but you do not report calibration or decision-curve analyses. Including these would meaningfully strengthen the clinical screening case and improve comparability.
  - Recent fairness audit frameworks (2607.28608) advocate multi-metric disparity (EOD), power-aware reporting (MDE), conditional permutation testing, and post-hoc mitigation. Your fairness section would benefit from adopting these, adding per-group calibration, and reporting statistical inference and support sizes.
  - Parsimonious feature-selection via tail dependence (2505.22554) reached AUC ≈0.823 on BRFSS with 10 features; this is highly relevant to your minimal-questionnaire result. A direct comparison (on the same splits) or a discussion would contextualize your SHAP-based reduction.
  - Some BRFSS studies report very high AUCs with heavy preprocessing and stacking (2512.02023), but concerns about leakage and inconsistent splits suggest caution; your multi-seed design and modest AUCs are more plausible and credible, but you should explicitly note safeguards against leakage (including handling of duplicates).
  - Survey-weighted modeling and conformal uncertainty for complex surveys (2403.19752) highlight the importance of design weights and UQ for population-level claims; incorporating survey weights and basic uncertainty sets would materially improve your claims about generalizability and screening safety.
- Discussion of broader impact and significance
  - The explicit positioning as a screening, hypothesis-generating tool is responsible and appropriate. Emphasis on interpretability and fairness is aligned with public health priorities.
  - Without survey weighting and stronger fairness diagnostics, deployment for equitable population screening remains premature; adding these elements, along with calibration and cost/benefit analyses (e.g., net benefit, decision curves), would provide a more complete, actionable picture.
  - The minimal-questionnaire insight is promising for low-resource settings; validating it externally and prospectively (and ensuring no selection bias in feature choice) could translate into practical screening tools.

## Questions

1. How did you ensure that exact duplicate feature profiles did not appear in both train and test folds? If they did, can you quantify their effect on performance by repeating experiments with de-duplication or grouped splitting?
2. Did you compute SHAP rankings for the minimal-questionnaire experiment on training data only? If rankings involved test data, can you re-run with nested CV or train-only SHAP to eliminate selection bias?
3. Can you report calibration metrics (Brier, ECE/MCE, calibration curves) overall and by subgroup, and discuss how calibration quality influenced your threshold optimization?
4. Please reconcile the inconsistency between the confusion-matrix recall (≈0.797) and the threshold table’s “Default” sensitivity (0.500). Which is correct, and how was the operating point chosen?
5. Given that BRFSS is a complex survey, do you plan to incorporate design weights in model training/evaluation and in fairness analysis? If so, how might you adjust your pipeline to support weight-aware learning and inference?
6. The age TPR gap (~0.82) is extremely large. Can you provide per-group supports, confidence intervals, and additional fairness metrics (e.g., equalized odds difference with inference, subgroup ECE/MCE)? What mechanisms (model vs. label/process bias) might explain this disparity?
7. For out-of-time validation, were variables harmonized across waves, and did you consider recalibration or threshold re-optimization per wave? Can you report performance variability and any drift diagnostics?
8. Could you compare your minimal feature set to alternative parsimonious selectors (e.g., relief-based or tail-dependence filters) and report whether SHAP-based ranking is consistently superior on your splits?

