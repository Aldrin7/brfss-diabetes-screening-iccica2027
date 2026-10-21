# Review (round 2): An Interpretable, Fairness-Aware Machine Learning Framework for Diabetes Screening from Population Health Surveys

Venue: ICCICA | Submitted: 2026-10-21T10:21:32.359046 | Reviewed: 2026-10-21T10:49:38.968209

1. Summary
The paper presents an interpretable, fairness-aware screening framework for diabetes using the 2015 BRFSS “Diabetes Health Indicators” dataset (n=253,680; 21 features), comparing logistic regression, random forest, and histogram-based gradient boosting with multi-seed evaluation. The study emphasizes reproducibility, calibration (with a held-out calibration split), fairness audits (with bootstrap CIs and support-aware analyses), out-of-time validation on BRFSS 2017–2021 waves, and a minimal-questionnaire experiment that preserves most discrimination using 5–11 SHAP-ranked features. Main findings include small but statistically significant AUC gains for gradient boosting over logistic regression, severe miscalibration from class weighting that is fixed post hoc, dominant predictive signal from clinical-history features, and an age-related TPR disparity that shrinks substantially under minimum-support constraints.

2. Strengths
- Technical novelty and innovation
  - Introduces a comprehensive reliability-first pipeline that pairs discrimination with calibration, leakage auditing, and fairness analyses on a widely used public dataset.
  - Support-aware fairness analysis (minimum-positive-count sensitivity) usefully qualifies headline disparity metrics and aligns with emerging audit practices.
  - Scenario probes (bounded perturbations) are clearly framed as behavioral diagnostics rather than causal claims.
- Experimental rigor and validation
  - Multi-seed reporting with paired bootstrap tests for AUC differences; ablations by feature groups; grouped CV and deduplicated refits to quantify duplicate-profile effects.
  - Out-of-time validation across later BRFSS waves with threshold re-optimization for screening sensitivity.
  - Post-hoc recalibration on a held-out split with both isotonic and Platt scaling; subgroup calibration summaries are provided.
- Clarity of presentation
  - Transparent articulation of validity boundaries (screening vs diagnosis; association vs causation) and limitations.
  - Clear, actionable deployment guidance (operating-point selection for sensitivity) and an interpretable baseline that is competitive with tree ensembles.
  - Code and run-manifest availability, pinned software versions, and multi-seed configuration support reproducibility.
- Significance of contributions
  - Demonstrates that a simple, interpretable baseline is near-parity with stronger ensembles for large-scale screening, which is pragmatic for public-health use.
  - Highlights calibration as a precondition for communicating probabilities; shows fairness gaps can be support-driven and mitigated via policy (e.g., stratified thresholds).
  - Minimal-questionnaire results are valuable for low-resource screening contexts, with performance consistent with recent feature-selection literature.

3. Weaknesses
- Technical limitations or concerns
  - Complex survey design is not incorporated (no weights/PSU/strata), so population-level performance and fairness estimates may be biased (see SaML guidelines).
  - “Fairness-aware threshold optimization” is largely threshold tuning for sensitivity with subgroup reporting; the “fairness-aware” label overstates the methodological contribution.
  - Some internal inconsistencies (e.g., duplicate counts vary: 24,206 vs 25,772; duplicate-leakage AUC deltas reported as both ~0.0001 and ~0.0008) undermine confidence.
  - Mixing prediabetes with diabetes in a single positive label changes prevalence/operating characteristics and complicates comparisons to studies with stricter case definitions.
- Experimental gaps or methodological issues
  - Decision-curve analysis (net benefit) and AUPRC are not reported despite class imbalance and screening context (cf. related work that foregrounds DCA).
  - Calibration choices could be broadened (e.g., Beta, Venn-Abers, or logit-Platt/GAM variants), and it is unclear if the same 2015 calibrator was reused for later waves.
  - The out-of-time validation does not report subgroup calibration/fairness drift or survey-weighted estimates; threshold re-optimization alone may be insufficient.
  - The held-out test split may still suffer from duplicate-profile leakage unless explicitly grouped; the leakage audit focuses on CV and a deduplicated refit rather than the final test protocol.
- Clarity or presentation issues
  - Occasional equation artifacts and table formatting glitches aside, some methodology details are ambiguous (e.g., exact “histogram-based GB” implementation vs XGBoost reference; calibration transfer across years).
  - The fairness section focuses on TPR/FPR gaps but could better connect to decision thresholds per subgroup and actionable remediation strategies.
- Missing related work or comparisons
  - Limited engagement with survey-aware ML and design-based calibration/validation (e.g., Matabuena et al. 2024; SaML guidelines) for population-representative claims.
  - Calibration literature that cautions about isotonic/Platt degradation and proposes alternatives (e.g., Beta, Venn-Abers, Platt-with-logit/GAM) could be discussed and tested.
  - Comparison to recent BRFSS pipelines reporting very high AUCs is thoughtful, but a side-by-side reimplementation (or stronger ablation) would strengthen the critique.

4. Detailed Comments
- Technical soundness evaluation
  - The modeling choices are sensible for tabular survey data, and the fairness/calibration emphasis is commendable. The explicit admission that balanced class weights distort probability scales—and the empirical demonstration that post-hoc calibration dramatically reduces ECE—are valuable contributions.
  - However, the lack of survey design integration is a core limitation for claims about population-level discrimination, calibration, and subgroup gaps. Recent work shows that ignoring weights/strata/PSUs biases estimates and misstates uncertainty and fairness gaps. Adopting survey-aware training/validation (or, at minimum, survey-weighted evaluation with design-based variance) would materially improve inference quality.
  - The “fairness-aware threshold optimization” appears to be global threshold tuning to meet sensitivity targets, with subgroup TPR gap reporting. If fairness is an objective, consider explicit constrained optimization (e.g., minimizing false negatives subject to bounded TPR gaps) or subgroup-specific thresholds with per-group calibration, as recommended in fairness-in-healthcare audits.
- Experimental evaluation assessment
  - Strengths: multi-seed reporting; paired bootstrap comparisons; ablations; SHAP explanations; scenario probes; out-of-time validation. The calibration pre/post analysis is thorough and clearly shows the necessity of post-hoc recalibration.
  - Gaps: no AUPRC or decision-curve analysis (important for class-imbalanced screening utility); unclear whether a 2015-derived calibrator was transported to 2017–2021 or refit per wave; per-wave subgroup calibration/fairness not reported; no DeLong tests for AUC differences; and no survey-weighted evaluation.
  - Duplicate-profile leakage: the grouped CV and deduplicated refits are appropriate diagnostics, but the final held-out test split should also enforce grouping (or use a deduplicated test) to fully exclude residual leakage from performance estimates.
- Comparison with related work (using the summaries provided)
  - Calibration: The finding that class weighting harms calibration aligns with broader evidence that post-hoc calibration is necessary and that method choice matters. Recent benchmarks show Beta calibration and Venn-Abers can outperform isotonic/Platt on average and avoid extreme degradations; the paper could broaden beyond isotonic/Platt and consider Platt-with-logit or GAM variants shown to be robust after sampling distortions.
  - Survey-aware ML: SaML (nine-step guideline) and survey-weighted NN/conformal work argue that design integration is essential for population-level claims. The manuscript acknowledges this but does not implement design-based inference; doing so would materially strengthen the study.
  - NHANES non-invasive screening: The reported AUCs (≈0.82) and SHAP-driven interpretability are consistent with recent NHANES-based models that emphasize calibrated outputs and DCA. Adding AUPRC and DCA would facilitate direct utility comparisons.
  - Feature selection: The minimal-questionnaire AUCs are consistent with copula tail-dependence feature selection (≈0.823 with k=10). The side-by-side comparison with MI/mRMR-like filters is helpful and fairly presented.
- Discussion of broader impact and significance
  - The paper’s practical orientation—interpretable baseline, robust calibration, explicit deployment caveats, and minimal-questionnaire variants—addresses real needs for population screening and outreach. The support-aware fairness analysis and emphasis on calibration are especially timely and responsible.
  - Without survey-aware evaluation and subgroup analyses that include race/ethnicity (not present in the cleaned subset), policy conclusions about health equity should remain cautious. The paper’s explicit ethical scope and non-diagnostic positioning are appropriate and appreciated.

5. Questions for Authors
1. Did you enforce grouped splitting for the final held-out test set (not only for CV), or alternatively evaluate on a deduplicated test set? Please clarify the exact protocol to conclusively rule out duplicate-profile leakage in the reported test metrics.
2. For the out-of-time validation, were the calibration maps (isotonic/Platt) refit on a per-wave calibration split, or was the 2015-derived calibrator transported to later waves? Please report per-wave calibration metrics (Brier, ECE/MCE) and subgroup calibration drift.
3. Why label prediabetes and diabetes jointly as positive? Can you provide sensitivity analyses separating these conditions (and report how this affects AUC, calibration, and subgroup gaps)?
4. Given known limitations of isotonic/Platt, did you evaluate alternative calibrators (e.g., Beta, Venn-Abers, Platt-with-logit or GAM)? If so, how did they perform; if not, could you add a brief comparison to ensure robustness?
5. Can you reconcile the differing counts and effects reported for duplicate profiles (24,206 vs 25,772; AUC deltas of ~0.0001 vs ~0.0008)? A consolidated table with one consistent leakage analysis would help.
6. How stable are the SHAP top features across seeds and across out-of-time waves? Please provide a stability metric (e.g., Jaccard overlap of top-k lists).
7. Would you consider survey-weighted evaluation (and design-based uncertainty) following SaML guidelines, at least for the final test estimates and subgroup gaps, to strengthen population-level inference?

6. Overall Assessment
This is a careful and practice-oriented screening study that prioritizes interpretability, calibration, and fairness diagnostics over chasing marginal AUC gains. The empirical work is generally rigorous (multi-seed reporting, calibration split, out-of-time validation, ablations, leakage checks) and well contextualized, and the minimal-questionnaire result is both practical and consistent with recent literature. The main limitation—lack of survey-aware methodology for performance, uncertainty, and fairness evaluation—meaningfully constrains population-level claims and should be addressed to meet a top-tier bar. Additional improvements include expanding calibration baselines, clarifying duplicate-handling in the final test protocol, reporting AUPRC and decision-curve analyses, and tightening some internal inconsistencies. With these revisions, the paper would offer a strong, reproducible, and ethically responsible contribution of clear utility to the ICCICA community.

TRIPLE_SCORES:
- Claims_Support: 0  # Strong empirical work, but survey-design omission and some ambiguities reduce the strength of support for population-level and fairness claims
- Experimental_Soundness: 0  # Solid core methodology and audits; key gaps: survey design not integrated, limited calibration baselines, DCA/AUPRC missing
- Writing_Clarity: 0  # Generally clear and well-structured, but some inconsistencies and ambiguous details should be fixed
- Prior_Work_Context: 0  # Related work is discussed thoughtfully; could incorporate survey-aware ML and broader calibration literature more fully
- Question_Importance: +1  # High-importance public-health screening problem with direct practical relevance
- Originality: 0  # Methodological assembly is careful and useful but not fundamentally novel; the support-aware fairness analysis is a nice touch
- Value_to_Community: 0  # Of moderate-to-high interest; with survey-aware evaluation and added calibration/utility analyses, value would be strong