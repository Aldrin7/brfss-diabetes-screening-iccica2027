# Review (round 3, two-column PDF): An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys

Venue: ICCICA | Submitted: 2026-10-21T10:44:33.897827 | Reviewed: 2026-10-21T11:06:14.455386

1. Summary
The paper presents a transparent, reproducible, and fairness-aware framework for diabetes screening using the public BRFSS 2015 “Diabetes Health Indicators” dataset (n=253,680, 21 features). It compares logistic regression, random forest, and histogram-based gradient boosting across five seeds, evaluates discrimination, calibration, and subgroup fairness, performs leakage and calibration audits, runs out-of-time validation on 2017–2021 BRFSS waves, and explores a reduced “minimal questionnaire” using SHAP-ranked features. The central empirical finding is that gradient boosting modestly outperforms logistic regression in AUC (0.8288 vs. 0.8212), but the gap is statistically significant and practically small; the authors argue for a screening-level logistic baseline, emphasize post-hoc recalibration after class weighting, and provide support-aware fairness reporting and threshold optimization for target sensitivity.

2. Strengths
- Technical novelty and innovation
  - The work thoughtfully integrates a suite of diagnostic elements—duplicate-profile leakage audit, post-hoc calibration analysis with strong improvements, support-aware fairness inference with bootstrap intervals, and bounded perturbation probes—to elevate methodological rigor beyond typical baselines on this dataset.
  - The support-restricted fairness reporting (minimum positive counts per stratum) is a practically meaningful refinement that reduces spurious disparities from sparse subgroups.
  - The “minimal questionnaire” experiment links explainability (SHAP) to actionable model simplification with marginal AUC loss, advancing usability for low-resource settings.
- Experimental rigor and validation
  - Multi-seed evaluation with fixed environments and pinned software versions, paired bootstrap AUC comparisons, and out-of-time validation on later BRFSS waves substantiate the reported claims and reproducibility.
  - Clear calibration audit demonstrating the miscalibration induced by class weighting and the corrective effect of isotonic/Platt scaling; the requirement to recalibrate before communicating probabilities is explicit.
  - Duplicate-profile leakage is tested via grouped CV vs. stratified CV and dedup refits, showing negligible impact—addressing a common source of inflation on this dataset.
- Clarity of presentation
  - The scope, limitations, and intended use (screening-level, not diagnostic/causal) are candid and consistently restated.
  - The SHAP analysis is positioned as descriptive (non-causal), and ablations connect feature groups with predictive value in an interpretable manner.
- Significance of contributions
  - Diabetes screening from open surveys is important, and the framework’s emphasis on calibration, fairness, and reproducibility is aligned with current best practices.
  - The out-of-time validation and threshold optimization guidance materially improve the translational value of the work for population screening programs.

3. Weaknesses
- Technical limitations or concerns
  - The analysis is not survey-aware: BRFSS’s complex design and weights are not incorporated in training, model selection, or evaluation, limiting validity of population-level claims and fairness assessments.
  - The label merges prediabetes and diabetes, which is pragmatic for screening but blurs clinical distinctions; some predictors (e.g., self-reported general health, high BP, difficulty walking) can be consequences rather than antecedents of diabetes, complicating interpretability for early detection.
  - The fairness work is primarily auditing; there is no implemented mitigation beyond threshold selection, and race/ethnicity is absent in this cleaned subset, reducing equity scope.
- Experimental gaps or methodological issues
  - Calibration is demonstrated on the 2015 split; it is unclear whether probabilities were recalibrated for later waves during temporal validation, or how calibration drift is handled over time.
  - PR-AUC and decision-curve/net-benefit analyses are not reported, which are relevant given class imbalance and screening use-cases.
  - The SHAP global explanations are computed on a 2,000-row test subsample; stability across the full test distribution (and per subgroup) is not systematically quantified beyond brief remarks.
- Clarity or presentation issues
  - There are inconsistencies in reported duplicate counts and in the age TPR gap reductions under minimum support; some tables appear garbled (likely extraction artifacts), obscuring precise values.
  - The calibration section briefly mentions MCE monotonicity statements and probability shifts without always providing full numerical context or confidence intervals.
- Missing related work or comparisons
  - Survey-aware ML (e.g., SaML guidelines) is discussed in limitations but not operationalized; a small weighted-vs-unweighted experiment would strengthen claims.
  - Broader calibration literature (e.g., Beta calibration, Venn-Abers; and local calibration error for individual reliability) is not evaluated despite being pertinent to fairness and deployment.
  - Recent multi-disease, calibrated screening frameworks and NHANES-based noninvasive screening studies are mentioned but could be used as stronger comparison points (e.g., decision-curve analysis, subgroup calibration/reliability).

4. Detailed Comments
- Technical soundness evaluation
  - The modeling choices (LR, RF, HGB) are appropriate for tabular survey data; the use of class weights is standard but correctly flagged as harmful to probability calibration absent post-hoc correction.
  - Leakage audit via grouped CV with duplicate-profile isolation is a valuable check; adopting grouped CV as the default during model selection would fully close this loop.
  - The fairness metrics (TPR gap, FPR gap) with bootstrap intervals and minimum support constraints are well-motivated, though the large age disparity deserves further analysis (e.g., age-specific thresholds or separate calibrations).
  - The work remains observational and cross-sectional; the authors appropriately avoid causal claims and frame “scenario probes” as behavioral sensitivity analyses.
- Experimental evaluation assessment
  - Multi-seed reporting with small standard deviations suggests stable estimates. Paired bootstrap for AUC differences is appropriate; consider also DeLong tests for completeness.
  - Calibration results (ECE from 0.2367 to 0.0031 with isotonic) are compelling; however, given known tradeoffs, it would be informative to compare Beta calibration and Venn-Abers (related work indicates different behaviors for strong tree ensembles).
  - The temporal validation is a strong addition. Clarify whether thresholds were re-optimized only for sensitivity or whether recalibration was also performed per wave; present ECE/Brier per wave with/without recalibration to assess drift.
  - Given screening context and class imbalance, include PR-AUC and decision-curve analysis; the latter can help justify operating thresholds by net benefit across plausible risk preferences.
- Comparison with related work
  - The paper situates its AUC results conservatively relative to prior BRFSS papers and plausibly attributes near-0.99 AUCs to methodological leakage or resampling artifacts; your leakage audit and seed stability support this position.
  - Compared to NHANES-based dysglycemia screening (AUC ≈ 0.82) and CardioMeta-style calibrated screening frameworks, the present work has similar discrimination but places greater emphasis on transparent audits for one disease and one survey, while lacking survey-weighted training/inference and multi-disease scope.
  - SaML-style survey integration (weights, design-based cross-validation) is identified as future work; even a small ablation (weighted evaluation only) would strengthen claims and better align with survey methodology literature.
  - Calibration literature: recent benchmarks show isotonic/Platt can sometimes degrade proper scoring for strong tabular learners; a brief comparative calibration study (isotonic vs. Platt vs. Beta vs. Venn-Abers) would enhance robustness of the calibration recommendation. Also, local calibration error (LCE) could provide subgroup- or instance-level reliability assessment relevant to fairness.
- Discussion of broader impact and significance
  - Emphasis on transparency, reproducibility, and honest boundaries of validity is commendable and well-aligned with ethical AI in health. The reduced-questionnaire finding is practically relevant for community screening.
  - Absent race/ethnicity limits equity assessment; deploying such a tool without evaluating race-stratified reliability risks obscuring disparities. Incorporating raw BRFSS race/ethnicity and adopting age- and group-specific thresholds/calibration would materially improve equity readiness.
  - Not incorporating survey design weights currently precludes population-level claims; this should be addressed before informing policy or public-health resource allocation.

5. Questions for Authors
1. Temporal validation and calibration: Were probabilities recalibrated per wave (2017/2019/2021) or was the 2015 calibration map reused? Please report per-wave ECE/Brier with and without recalibration to quantify calibration drift over time.
2. Fairness reporting inconsistency: The text states the age TPR gap drops from ≈0.788 to ≈0.144 under minimum-support constraints, but the table shows larger residual gaps (e.g., ≈0.421 at min positives=100). Which figures are correct, and what is the exact support threshold for the 0.144 value?
3. Duplicate counts: The manuscript alternately mentions 24,206 and 25,772 duplicate profiles. Please reconcile these counts and specify the definition of “duplicate” used (exact feature vector equality on all 21 variables?).
4. Survey design: Can you provide a small supplementary analysis computing weighted AUC/ECE and weighted subgroup metrics using BRFSS design weights (at least for evaluation), to clarify how unweighted vs. weighted conclusions differ?
5. Calibration methods: Given recent evidence that isotonic/Platt can underperform on strong tabular models, did you attempt Beta calibration or Venn-Abers predictors? If so, how did ECE/Brier compare, and if not, can you justify the chosen calibrators?
6. Minimal questionnaire stability: How stable are the top-k SHAP-selected features across seeds and across the later BRFSS waves? Do top-5/top-8 sets remain similar, and how does AUC vary when features shift?
7. Threshold optimization and fairness: Beyond sex, how do income- and education-stratified TPR gaps change across your sensitivity-targeted thresholds? Did you consider age-specific thresholds to address the large age disparity?
8. Scenario probes: For the bounded perturbation probes (e.g., +5 BMI), can you report subgroup-specific effects and discuss whether monotonicity/linearity assumptions hold? Also, were probes computed on calibrated or raw scores?
9. Label definition: Since the target combines prediabetes and diabetes, did you analyze whether performance differs between these subgroups (e.g., sensitivity within diagnosed diabetes vs. prediabetes cohorts), and does the model unduly favor one?
10. Reproducibility: The repository link is provided; could you ensure all artifacts (fixed seeds, splits, calibration folds, grouped-CV scripts, and bootstrap code) are included and that SHA hashes/manifests allow exact reproduction?

6. Overall Assessment
This is a careful, transparent, and practically motivated study that advances the quality bar for work on the public BRFSS diabetes dataset. The combination of multi-seed evaluation, leakage and calibration audits, support-aware fairness analysis, and temporal validation provides a robust empirical foundation. The main scientific limitations—non-incorporation of BRFSS complex survey design, absence of race/ethnicity, and a primarily auditing (not mitigating) approach to fairness—temper the strength of policy- or deployment-facing claims. Nonetheless, the paper’s methodological hygiene, reproducibility, and clear articulation of boundaries add meaningful value to the literature, particularly as a trustworthy baseline and guidance on calibration and fairness reporting for survey-based screening. With clarifications (notably on fairness gaps and duplicates), stronger calibration comparisons, and at least weighted evaluation, the paper would be a solid fit for ICCICA. I recommend acceptance after revision focusing on survey-awareness in evaluation, calibration robustness, and resolving presentation inconsistencies.

TRIPLE_SCORES:
- Claims_Support: +1  # Are the central claims adequately supported with evidence?
- Experimental_Soundness: 0  # Are the experimental setup and research methodology sound?
- Writing_Clarity: 0  # Is the writing clear and well-organized?
- Prior_Work_Context: 0  # Is the work properly contextualized relative to prior work?
- Question_Importance: +1  # Are the research questions being asked important?
- Originality: 0  # Does the paper bring significant originality of ideas and/or execution?
- Value_to_Community: +1  # Are the results valuable to share with the broader ICCICA community?