# Review (round 6): What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for Diabetes from Population Health Surveys

Venue: ICCICA | Submitted: 2026-10-21T13:54:15.886338 | Reviewed: 2026-10-21T14:08:26.185842

1. Summary
The paper audits an interpretable screening framework for diabetes using the 2015 BRFSS “Diabetes Health Indicators” dataset (n=253,680; 21 features), focusing on leakage control, calibration, fairness, explainability, and deployment utility. Across logistic regression, random forest, and histogram-based gradient boosting, the authors show that headline AUCs depend on holdout protocols, class-weighting harms calibration without improving discrimination, attribute-specific thresholds fail to transfer out of time, and subgroup disparities—especially by age—are not resolved by thresholding. The study provides multi-seed evaluations, multiple held-out protocols (including deduplication), a detailed calibration audit with four calibrators, out-of-time validation (2017–2021), SHAP-based explanations with stability checks, and decision-curve analysis to support sensitivity-oriented operating points.

2. Strengths
- Technical novelty and innovation
  - The paper reframes model development as an audit, centering leakage control, calibration, and fairness over chasing AUC, and reports negative results that change practice (e.g., class weights hurt calibration; group thresholds don’t transport).
  - A three-protocol leakage audit and explicit reconciliation of duplicate definitions provide methodological clarity on a dataset known for leakage pitfalls.
  - The fairness analysis goes beyond point estimates by reporting subgroup supports, bootstrap CIs, design-effect sensitivity, and intersectional cells, clarifying where data are too sparse to support strong claims.
- Experimental rigor and validation
  - Multi-seed train/test splits, bootstrap inference for AUC differences and fairness gaps, and decision-curve net benefit are well-chosen and aligned with screening objectives.
  - Calibration is audited comprehensively (Brier, ECE, MCE), with four post-hoc methods compared and transportability of the 2015-fitted map assessed out-of-time.
  - Out-of-time validation across three later BRFSS waves is a meaningful test of temporal transport.
- Clarity of presentation
  - Clear articulation of the audit questions, what the audits changed, and validity boundaries (e.g., screening vs diagnostic use; association vs causation).
  - Transparent reporting of model settings, seeds, software versions, and code/manifests for reproducibility.
- Significance of contributions
  - Demonstrates that commonly reported near-perfect AUCs on this dataset are unlikely to be reproducible under leakage-aware protocols, a valuable corrective for the community.
  - Offers practical guidance: unweighted training with a sensitivity-targeted, global threshold; discourages subgroup-specific thresholds without strong external validation.
  - The stability analysis of SHAP attributions and counterfactual flip counts provides nuanced interpretability insights for practice.

3. Weaknesses
- Technical limitations or concerns
  - The analysis does not incorporate survey design (weights, PSUs, strata), limiting population-level inference and possibly fairness conclusions; this is a major limitation for BRFSS-derived claims.
  - The label-noise “refit” that removes contradictory-label duplicates may induce optimistic performance and is not obviously aligned with deployable practice.
- Experimental gaps or methodological issues
  - Temporal validation remains within BRFSS; there is no cross-survey external validation (e.g., NHANES) where label definitions and measurement differ.
  - The grouped held-out protocol yields a slightly higher AUC than the plain split while the deduplicated test lowers AUC; the paper could more explicitly reconcile these seemingly counterintuitive movements and their implications.
  - Fairness mitigation exploration stops at per-attribute thresholds and per-group calibration; other portable mitigations (e.g., robust threshold selection, drift-aware or equalized-odds post-processing over time, distributionally robust training) are not evaluated.
- Clarity or presentation issues
  - There are nontrivial formatting artifacts in tables/figures. One SHAP bar chart appears to reference features not present in BRFSS (e.g., “ever_married,” “work_type”), which risks confusing readers and undermines interpretability claims.
  - Some metric choices and contexts vary (e.g., fairness at threshold 0.5 versus at the deployed sensitivity-oriented threshold), making cross-table comparisons difficult.
- Missing related work or comparisons
  - While the paper references calibration literature and recent BRFSS work, it does not fully engage with survey-aware ML frameworks (e.g., SaML guidelines) or design-based learning as a baseline approach for population claims, despite discussing them as limitations.
  - Claims about published pipelines reporting AUC near 0.99 would benefit from explicit replication attempts or a more detailed methodological critique of representative studies.

4. Detailed Comments
- Technical soundness evaluation
  - The audit framing is technically sound and appropriate for this dataset: three leakage-aware protocols, deduplication, label-noise sensitivity, and multi-seed reporting close common failure modes.
  - The calibration audit is careful: the strong finding that class weights severely distort the probability scale is consistent with broader benchmarks and theory; the transport analysis of isotonic mapping across waves is useful and pragmatic.
  - The fairness analysis correctly emphasizes supports and uncertainty, and the documented non-transfer of attribute-specific thresholds out of time is a valuable negative result.
- Experimental evaluation assessment
  - Metrics are well-chosen for screening (AUC, AUPRC, Brier, ECE, DCA), and paired bootstraps for AUC differences are appropriate given the modest separations across models.
  - Out-of-time validation within BRFSS shows mild degradation and acceptable calibration transport, but conclusions about equity and transportability would be stronger with cross-survey external validation (e.g., NHANES) where prior work observed larger drops.
  - The decision-curve analysis substantiates the recommended operating region (<0.5 threshold) and provides actionable guidance.
- Comparison with related work (using the summaries provided)
  - The observation that post-hoc calibrators can help or harm depending on base-model score properties aligns with large-scale benchmarks (2601.19944); the finding that Venn–Abers underperforms here is plausibly due to misaligned assumptions under severe miscalibration from class weighting, consistent with that benchmark’s cautions.
  - Relative to prior BRFSS studies reporting very high AUCs (e.g., 2512.02023), this paper’s leakage-aware protocols and deduplication provide credible evidence that such results likely reflect leakage/resampling artifacts rather than real generalization.
  - The lack of survey weighting and design-aware inference contrasts with SaML guidance (2605.08963) and design-based ML for surveys (2403.19752); integrating these would materially strengthen population-level claims and fairness conclusions.
  - The NHANES-focused works (2606.16056; 2607.16253) highlight cross-dataset calibration and fairness challenges; this paper’s out-of-time but within-survey validation is a useful intermediate step but does not substitute for cross-survey tests.
- Discussion of broader impact and significance
  - The paper usefully resets expectations for this widely used dataset, promoting better practice around leakage audits, calibration, and fairness evaluation with uncertainty and supports.
  - The negative results (no benefit from class weighting; subgroup thresholds not portable) are impactful for deployers of screening tools and likely to influence community baselines.
  - The lack of survey-aware analysis currently limits policy relevance; adopting design-based methods would enable population-representative performance and fairness statements.

5. Questions for Authors
1. Can you clarify whether the calibration-comparison table (isotonic/Platt/Beta/Venn–Abers) is based on the class-weighted model only? If yes, could you also report calibration metrics for the unweighted base (which you recommend for deployment) to avoid misinterpretation?
2. The SHAP bar chart that lists features such as “ever_married” and “work_type” appears inconsistent with BRFSS Diabetes Health Indicators. Could you correct this figure and ensure all interpretability plots correspond to the 21-feature BRFSS subset?
3. For the grouped held-out protocol (P3), AUC increases slightly relative to the plain split while the deduplicated test (P2) reduces AUC. Can you elaborate on why P3 ends up higher and how fold construction might advantage the model despite stricter grouping?
4. How sensitive are the fairness gaps to the chosen operating threshold in the range supported by your decision-curve analysis? Could you provide fairness summaries at the recommended sensitivity-targeted threshold rather than 0.5 for easier alignment with deployment?
5. Could you report results from at least one survey-aware evaluation (e.g., reweighting metrics by approximate BRFSS weights or a sensitivity analysis to plausible design effects) to gauge how much your core conclusions might change under design-based inference?
6. On the label-noise “refit” where contradictory-label duplicates are removed: how would you operationalize this in a prospective setting, and what are the risks of overfitting to artifacts that may not be resolvable at deployment time?
7. Did you consider portable fairness post-processing (e.g., equalized-odds post-processing with calibration per group) or thresholding schemes that explicitly hedge against prevalence drift across time? If yes/no, please discuss expected trade-offs.

6. Overall Assessment
This is a thoughtful, well-executed audit of diabetes screening models on BRFSS that prioritizes reproducibility, calibration, and fairness over headline discrimination. The work corrects misleadingly high AUC claims reported elsewhere on this dataset, demonstrates that class weighting can be actively harmful for calibration, and provides a valuable negative result that attribute-specific thresholds do not transfer out of time. The study’s strengths lie in its methodological care (leakage audits, calibration analysis, decision-curve utility) and transparent reporting with released artifacts. The main limitations are the absence of survey-aware inference, the lack of cross-survey external validation, and some presentation errors (notably a mismatched SHAP figure) that should be fixed. With revisions addressing the clarity issues and, ideally, adding a design-aware evaluation or sensitivity analysis, this paper would offer strong value to the ICCICA community. As submitted, I view it as a solid contribution with meaningful practical impact, meriting acceptance after minor-to-moderate revision.

TRIPLE_SCORES:
- Claims_Support: [+1]  # Are the central claims adequately supported with evidence?
- Experimental_Soundness: [0]  # Are the experimental setup and research methodology sound?
- Writing_Clarity: [0]  # Is the writing clear and well-organized?
- Prior_Work_Context: [0]  # Is the work properly contextualized relative to prior work?
- Question_Importance: [+1]  # Are the research questions being asked important?
- Originality: [+1]  # Does the paper bring significant originality of ideas and/or execution?
- Value_to_Community: [+1]  # Are the results valuable to share with the broader ICCICA community?