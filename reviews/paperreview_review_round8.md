# Review (round 8): What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for Diabetes from Population Health Surveys

Venue: ICCICA | Submitted: 2026-10-21T15:48:51.026477 | Reviewed: 2026-10-21T16:15:09.215329

1. Summary
This paper audits an interpretable, fairness-aware screening framework for diabetes using the 2015 BRFSS-derived “Diabetes Health Indicators” dataset (n = 253,680; 21 features). It contributes a set of leakage audits (including duplicate reconciliation), rigorous calibration analyses (showing class-balancing weights severely harm calibration), fairness quantification with bootstrap intervals and group supports, and a negative-result finding that attribute-specific thresholds fitted on one wave do not transfer out of time. The work emphasizes reproducible, deployment-relevant evaluation (global sensitivity-targeted thresholds, calibration transport, decision-curve net benefit), and documents where the dataset cannot support population-level claims due to absent design variables and race/ethnicity.

2. Strengths
- Technical novelty and innovation
  - Introduces a clear, three-protocol leakage audit with reconciled duplicate definitions (identical full rows vs. identical covariate vectors), quantifying contradicting-label duplicates and bounding leakage-induced optimism.
  - Provides a careful calibration audit across multiple calibrators (isotonic, Platt, Beta, Venn–Abers) and demonstrates a strong, practice-relevant negative result: class-balancing weights inflate ECE by ~60× without improving discrimination.
  - Conducts a fairness audit with bootstrap CIs, explicit subgroup supports, design-effect sensitivity, and support-restricted estimates, improving interpretability of gap metrics.
  - Offers an empirically supported negative result that attribute-specific thresholds fitted on one wave fail to transfer temporally; proposes a stable global sensitivity-targeted threshold instead.
- Experimental rigor and validation
  - Multiple seeds with full re-splits, paired bootstraps for model comparisons, per-wave (2017/2019/2021) out-of-time validation, and decision-curve analysis for clinical utility.
  - Leakage controls include a deduplicated test protocol and a grouped held-out scheme ensuring identical covariate vectors do not cross folds.
  - SHAP importance with stability checks across seeds/waves and permutation-importance cross-validation; feature-group ablations and reduced-form analyses.
- Clarity of presentation
  - Careful delineation of operating points and the use of raw vs recalibrated scores; explicit thresholds and calibration transport analyses.
  - Precise reporting of metrics (ROC-AUC, AUPRC, ECE, Brier, net benefit) with uncertainty quantification and subgroup supports.
  - Transparent discussion of limitations (no design variables; race/ethnicity unavailable; collapsed prediabetes/diabetes).
- Significance of contributions
  - Provides a reproducible screening-oriented evaluation that directly challenges unreproducible AUC≈0.99 claims on the same dataset and explains likely causes.
  - Offers concrete, audit-driven guidance for practitioners: avoid class weights when calibrated probabilities are required; use a single global sensitivity-targeted threshold; do not rely on subgroup-specific thresholds unless validated out of time.

3. Weaknesses
- Technical limitations or concerns
  - Absence of survey design variables (weights, PSUs, strata) prevents population-level inference; although acknowledged, this constrains fairness interpretation and external validity.
  - Race/ethnicity is missing, limiting fairness conclusions to a subset of socially salient groups; age disparities may interact with unobserved race effects.
  - The contradictory-label duplicate analysis risks over-attributing inconsistency to the labels rather than recognizing distinct individuals can share identical covariates but differ in outcome.
- Experimental gaps or methodological issues
  - No external, cross-survey validation (e.g., NHANES) despite related literature showing notable transfer degradation; this restricts claims about transportability beyond BRFSS.
  - The Venn–Abers implementation and prior-correction assumption are only qualitatively explained; a more formal description would aid reproducibility.
  - The randomized hyperparameter search budget is modest; while not harming the central claims, stronger tuning would dispel concerns that small differences are configuration-bound.
- Clarity or presentation issues
  - Table IV shows the sex-specific policy yields the same sex TPR gap as the global policy in-sample, which reads as redundant without explanation (sex gap was already small; still worth clarifying).
  - The paper mixes raw and calibrated scores across some fairness and operating-point analyses; while justified, a small schematic could help readers track which pipeline stage each result references.
- Missing related work or comparisons
  - The survey-aware ML guideline literature (e.g., SaML) could be cited more explicitly to contextualize design-based inference needs and gaps.
  - A brief comparison to multi-task or cross-domain screening frameworks (e.g., CardioMeta-style) could underscore the paper’s distinctive contribution as an audit-first, single-task BRFSS study.

4. Detailed Comments
- Technical soundness evaluation
  - The leakage audit is well-motivated and carefully executed. Distinguishing strict from covariate-only duplicates and quantifying contradicting-label pairs is a strong diagnostic of potential label noise and leakage risk. The three held-out protocols triangulate the leakage impact convincingly (≤0.01 AUC).
  - Calibration work is meticulous: the demonstration that class weights distort the probability scale while not improving discrimination is important for screening tools. The ranking of calibrators (Beta ≲ isotonic ≲ Platt; Venn–Abers worst here due to violated assumptions) is plausible and consistent with known behavior on tabular models.
  - The fairness audit properly accompanies gap metrics with subgroup supports and bootstrap CIs; the support-restricted age estimate is a commendable practice that avoids over-interpreting sparse cells. Design-effect sensitivity is a thoughtful addition given missing survey variables.
  - The negative result on subgroup thresholds not transferring out of time is carefully diagnosed (prevalence and score-shift by level, not threshold drift) and is a valuable contribution.
- Experimental evaluation assessment
  - Multi-seed reporting, paired bootstraps, and out-of-time validation across three subsequent waves support robustness. Decision-curve analysis connects model operation to plausible clinical trade-offs.
  - SHAP stability across seeds and waves, along with permutation-importance agreement and counterfactual flip checks, demonstrates due diligence. The flip analysis appropriately cautions against reading SHAP as leverage for binary variables.
  - The out-of-time calibration-transport experiment (small ECE penalties) strengthens deployment claims for a global threshold, but per-group transport remains challenging (core finding on subgroup thresholds).
- Comparison with related work (using the summaries provided)
  - Relative to NHANES-focused dysglycemia screening (e.g., 2606.16056), this study is BRFSS-based, non-invasive, and audit-first; both emphasize calibration and decision-curve net benefit. The present paper’s out-of-time within-survey validation complements prior NHANES studies that lack temporal holds or cross-survey external validation.
  - On survey design, this work aligns with 2403.19752 and the SaML guideline (2605.08963) in recognizing the need for design-aware modeling; here, the absence of design variables is candidly acknowledged and treated as a limitation, with sensitivity analyses for design effects.
  - Compared to feature-selection work (2505.22554), the SHAP-based reduced forms here achieve similar AUCs (≈0.823–0.825 with ≈10 features), suggesting convergent evidence that compact questionnaires preserve most signal.
  - The paper explicitly counters inflated AUC reports in BRFSS studies (see 2512.02023; and other claims approaching 0.99) and demonstrates through leakage audits and deduplication that such figures are unlikely to be reproducible under leakage-aware protocols—a valuable corrective for the community.
  - Multi-task frameworks (2607.15721) show calibration advantages and domain-shift sensitivity across datasets; this paper’s focused, single-task BRFSS audit complements that perspective by isolating leakage, calibration, and fairness pitfalls in a high-visibility public dataset.
- Discussion of broader impact and significance
  - The work provides concrete guardrails for public-health screening on open surveys: avoid class weights when probability calibration matters, do not over-interpret subgroup thresholds learned from a single wave, and report subgroup supports/intervals. These practices should raise the bar for reproducibility and equity audits in population-level screening research.
  - Limitations around absent design features and race/ethnicity are material for equitable deployment; the paper is responsibly cautious in claims and points to precise next steps (design-aware modeling on raw BRFSS XPT; cross-survey validation; out-of-time validated fairness mitigation).
  - The negative results are as valuable as positive ones here, particularly the failure-to-transfer of subgroup thresholds, which practitioners often assume will hold.

5. Questions for Authors
1. Please clarify the exact Venn–Abers variant and prior-correction details (objective, constraints) and provide the fitted calibration curves or parameters; could the unexpectedly poor performance be replicated on an unweighted base model for completeness?
2. In Table IV, the sex-specific policy shows the same in-sample sex TPR gap as the global threshold; was this because the global gap was already small, or is there a reporting/rounding artifact? Could you add per-level sensitivities for each policy to illustrate the redistribution effect?
3. For the contradictory-label duplicates: have you examined whether these arise from truly duplicated respondents versus distinct individuals with identical covariates (which would not imply label error)? Any sensitivity restricting the duplicate definition to near-identical but not exactly identical vectors?
4. Could you report subgroup AUCs and per-group calibration (ECE/Brier) for 2015 and 2021 to complement TPR/FPR gaps, especially for the age groups? This would help disentangle discrimination vs. calibration contributions to disparities.
5. For the support-restricted age gap, what are the exact age-band definitions and positive counts per retained band? A small table in the appendix would aid reproducibility.
6. Since the global sensitivity-targeted threshold transports well, did you evaluate per-group recalibration (e.g., group-wise isotonic) before applying a single transported global threshold? If so, how did per-group calibration transport compare to overall calibration transport?
7. Could you provide net benefit stratified by key subgroups (e.g., age bands) to verify consistent clinical utility across groups at the recommended operating point?
8. Given your strong calibration findings against class weights, did you evaluate focal loss or asymmetric loss variants as an alternative to class weights, and if so, how did they affect calibration and net benefit?

6. Overall Assessment
This is a careful, well-executed audit paper that prioritizes reproducibility, calibration, fairness transparency, and deployment practicality over headline AUC gains. Its leakage audit and contradictory-label analysis provide a credible explanation for previously reported, unrealistically high AUCs on the same dataset. The calibration study and the negative-result finding that subgroup thresholds do not transfer out of time are particularly impactful and directly actionable for practitioners. Limitations—most notably the absence of survey design variables and race/ethnicity—constrain population-level and equity claims, and external cross-survey validation remains an important gap. Nonetheless, within its stated scope, the paper is technically sound, clearly presented, and offers valuable, practice-shaping insights for screening with public health surveys. I recommend acceptance after minor revisions focused on clarifying calibration implementation details, explicating subgroup-threshold results, and adding a brief connection to survey-aware ML guidelines.

```
TRIPLE_SCORES:
- Claims_Support: +1  # Are the central claims adequately supported with evidence?
- Experimental_Soundness: +1  # Are the experimental setup and research methodology sound?
- Writing_Clarity: +1  # Is the writing clear and well-organized?
- Prior_Work_Context: 0  # Is the work properly contextualized relative to prior work?
- Question_Importance: +1  # Are the research questions being asked important?
- Originality: 0  # Does the paper bring significant originality of ideas and/or execution?
- Value_to_Community: +1  # Are the results valuable to share with the broader ICCICA community?
```