# Review (round 5): What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for Diabetes from Population Health Surveys

Venue: ICCICA | Submitted: 2026-10-21T13:21:15.369765 | Reviewed: 2026-10-21T13:38:07.393358

1. Summary
This paper audits an interpretable, fairness-aware diabetes screening framework built on the 2015 BRFSS (n=253,680; 21 variables), focusing on the gap between headline accuracy and real deployment needs. The authors evaluate leakage (three protocols, duplicate/label-noise reconciliation), calibration (four calibrators, transport across waves), fairness (TPR/FPR gaps with bootstrap CIs and group supports), explainability stability (TreeSHAP), and utility (decision-curve analysis), alongside multi-seed model comparisons (LR, RF, HGB). Key findings include: (i) class-balancing weights substantially harm calibration without improving discrimination, (ii) attribute-specific thresholds equalize targeted gaps in-sample but fail to transfer out of time, and (iii) the headline AUC depends on leakage control, with honest estimates near 0.82–0.83 rather than previously reported ~0.99.

2. Strengths
- Technical novelty and innovation
  - A careful, multi-pronged leakage audit (plain split, deduplicated test, grouped held-out) that quantifies duplicate-profile and contradictory-label effects and reconciles common duplicate definitions.
  - A well-structured calibration audit across four methods, including transport of calibration to future BRFSS waves, and a clear demonstration that class-weighting severely distorts probability scales in this task.
  - Fairness analysis with bootstrap inference and explicit reporting of per-group supports, intersectional cells, and support-restricted estimates, emphasizing where data do and do not suffice.
  - Negative results of practical importance: attribute-specific thresholds do not generalize out of time; class weights are unnecessary and harmful for calibration; age disparities persist across operating points.
- Experimental rigor and validation
  - Multi-seed splits (5 seeds), cross-validation, paired bootstrap tests for AUC differences, and decision-curve net benefit analyses aligned with screening use-cases.
  - Out-of-time validation on multiple later BRFSS waves (2017, 2019, 2021) with transported calibration, reporting small transport penalties.
  - SHAP-based explainability with stability metrics across seeds and waves; complementary flip-count checks near decision thresholds.
  - Open, reproducible implementation with pinned versions and artifact hashes.
- Clarity of presentation
  - Clear articulation of research questions, practical requirements, and interpretive boundaries; candid limitations and validity boundaries are explicitly numbered and discussed.
  - Distinguishes discrimination from calibration and utility, and fairness gaps from base-rate effects; uses CIs and supports to avoid over-interpretation.
- Significance of contributions
  - Offers a reproducible, realistic picture of survey-based diabetes screening performance and pitfalls, directly addressing over-optimistic literature reports.
  - Provides actionable guidance for deployment (global sensitivity-targeted threshold; avoid class weighting) and for fairness practice (validate mitigations out of time, report supports and CIs).

3. Weaknesses
- Technical limitations or concerns
  - No incorporation of complex survey design (weights/PSUs/strata) means results are sample-level, not population-level; this is critical for BRFSS and fairness claims across socioeconomic strata.
  - Race/ethnicity is absent, limiting equity analysis and external validity for U.S. contexts where race-related disparities are central.
  - The outcome collapses prediabetes and diabetes, blurring severity distinctions and potentially conflating antecedents with consequences.
  - Explainability relies on TreeSHAP over a 2,000-row sample; while stability is reported, more comprehensive or alternative attribution (e.g., Banzhaf) could strengthen conclusions.
- Experimental gaps or methodological issues
  - Only three model families are compared with relatively light hyperparameter search; broader tabular leaders (e.g., LightGBM/XGBoost with tuned configs) could confirm the suggested “performance ceiling.”
  - The failure of attribute-specific thresholds out of time is reported but not deeply diagnosed (e.g., group prevalence/score distribution shifts, groupwise calibration drift).
  - The deduplication/label-noise removal result that “AUC would rise to 0.8397” needs clearer methodological exposition to reconcile with other leakage-adjusted estimates.
- Clarity or presentation issues
  - Some tables contain formatting artifacts (e.g., illegible segments; thresholds at 0.5 in one fairness table vs. sensitivity-targeted thresholds elsewhere), which complicates quick cross-referencing.
  - Venn–Abers implementation details are sparse; given the atypically poor performance reported, more methodological context would help.
- Missing related work or comparisons
  - Survey-aware ML for complex designs (e.g., SaML guidelines) is acknowledged but not operationalized; a design-weighted evaluation sensitivity check would materially strengthen claims.
  - Alternative explanation methods and explanation-stability literature could be engaged more deeply beyond SHAP (e.g., Banzhaf values and robustness).

4. Detailed Comments
- Technical soundness evaluation
  - The leakage audit is well-motivated and well-executed; distinguishing exact duplicates from covariate-duplicate-with-contradictory-labels is important and often overlooked.
  - The calibration study is careful, and the finding that class-weighting inflates predicted prevalence and ECE is persuasive. Matching calibrator choice to miscalibration mode is a valuable takeaway.
  - Fairness metrics with bootstrap CIs and supports align with best practice; the explicit support-restricted age-gap analysis is a commendable demonstration of “how much is support starvation?”
  - The decision-curve analysis appropriately centers clinical utility and supports sub-0.5 thresholds.
- Experimental evaluation assessment
  - Multi-seed procedures, paired bootstrapping for AUC differences, and out-of-time evaluation add credibility. The transport of an isotonic map across waves with small ECE penalties is a useful practical result.
  - The attribute-specific threshold experiment is a meaningful “what if” audit; the strong degradation out of time (e.g., income gap expansion) is a sober and valuable negative result.
  - However, the absence of design-aware training/evaluation is a major limitation for BRFSS; even a sensitivity check with survey weights would improve the inferential framing for subgroup gaps and calibration.
  - The model suite could be slightly broadened (e.g., tuned LightGBM/XGBoost) to ensure the “practically negligible” gap vs LR is not a function of conservative tuning.
- Comparison with related work (using the summaries provided)
  - The critique of BRFSS studies reporting ~0.99 AUC (e.g., stacked ensembles with potential leakage/SMOTE misplacement) is consistent with the related work that has missing leakage safeguards and inconsistent splits; this paper’s leakage-audited AUC ~0.83 is credible by comparison.
  - Relative to NHANES non-invasive dysglycemia (AUC ~0.82), the discrimination is similar, reinforcing a likely performance ceiling with non-lab features; the decision-curve emphasis in both papers is aligned.
  - Against EHR-based long-horizon forecasting (AUC ~0.80 on richer features), this study’s cross-sectional screening task on self-reports reasonably lands in the same performance band.
  - SaML (survey-aware ML) emphasizes design-based evaluation and splitting; the current paper cites design-aware remedies but does not implement them—this is the largest gap to address for population-level or equity claims.
- Discussion of broader impact and significance
  - The work provides concrete, reproducible guidance that should recalibrate expectations for BRFSS-like screening models and help avoid misleadingly high AUC claims.
  - The negative results on subgroup thresholds and class weighting are important for fairness-focused deployment pipelines: validate mitigations out of time; avoid weighting when it breaks probability scales.
  - By documenting precisely where data cannot support population-level claims, the paper models responsible auditing practice.
- Additional presentation and reproducibility notes
  - The open code, pinned dependencies, and artifact hashing meaningfully enhance reproducibility.
  - Cleaning minor table artifacts and clarifying threshold choices per table would further improve readability.

5. Questions for Authors
1. You state that removing rows with contradictory labels for identical covariate vectors would raise AUC to 0.8397. Can you detail the exact protocol (train/test split, deduplication scope, seeds) and reconcile this increase with the deduplicated-test estimate of 0.8177?
2. Can you provide a survey-weighted evaluation (e.g., Horvitz–Thompson-style AUROC and ECE) as a sensitivity analysis to bound how much the unweighted results might deviate from population-level estimands?
3. What diagnostics explain the failure of attribute-specific thresholds out of time (e.g., groupwise calibration drift, prevalence shift, shifts in score distributions)? Would groupwise calibration transport or prevalence-adjusted thresholding improve stability?
4. Please clarify your Venn–Abers implementation (library, settings) and why it underperformed here relative to prior benchmarks. Did you try variants (e.g., inductive Venn–Abers) or different calibration folds?
5. SHAP analyses were computed on a 2,000-row test sample. Do the global rankings and stability metrics remain consistent when computed on the full test set, and did you consider Banzhaf values as a robustness check?
6. Table 7 reports fairness at threshold 0.5 whereas operational analyses use sensitivity-targeted thresholds near 0.115. Could you add fairness metrics at the deployed operating point to ensure consistent interpretation?
7. For the large age disparity, besides support restriction, did you assess whether disparities persist within strata of non-modifiable risk (e.g., conditioning on BMI/GenHlth) to separate prevalence-driven from model-driven effects?
8. Given your findings on class-weighting and calibration, would training with focal loss or calibrated cost-sensitive objectives (rather than sample weighting) avoid the ECE blow-up while preserving sensitivity?

6. Overall Assessment
This is a careful, well-executed audit that brings needed clarity to survey-based diabetes screening on BRFSS. The paper’s central contributions are methodological hygiene (leakage audits, calibration transport, fairness with supports and CIs) and practically important negative results (class weighting harms calibration; subgroup thresholds do not transfer across time). The work is transparent about validity boundaries and offers reproducible code and artifacts. The main limitation is the lack of design-based training/evaluation, which constrains population-level and equity interpretations in a complex-survey setting; race/ethnicity is also absent. Despite these caveats, the study is valuable for the community: it sets a realistic performance bar, diagnoses common pitfalls behind inflated claims, and provides actionable deployment guidance. I recommend acceptance after addressing clarifications (especially around label-noise removal and threshold consistency) and, if feasible, adding a survey-weighted evaluation sensitivity check.

TRIPLE_SCORES:
- Claims_Support: [+1]  # Are the central claims adequately supported with evidence?
- Experimental_Soundness: [0]  # Are the experimental setup and research methodology sound?
- Writing_Clarity: [+1]  # Is the writing clear and well-organized?
- Prior_Work_Context: [+1]  # Is the work properly contextualized relative to prior work?
- Question_Importance: [+1]  # Are the research questions being asked important?
- Originality: [0]  # Does the paper bring significant originality of ideas and/or execution?
- Value_to_Community: [+1]  # Are the results valuable to share with the broader ICCICA community?