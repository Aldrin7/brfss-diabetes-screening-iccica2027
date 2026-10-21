# Review responses — round 6 (internal record)

Paper: "What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for Diabetes from Population Health Surveys"
Reviewer round: PaperReview.ai round 6 (submitted 2026-10-21)
Status: all seven reviewer questions and the six stated weaknesses are addressed in the
6-page IEEEtran build (`archive/ieeetran_2col/`, 6 pages each;
anonymized 6 pages). Every number below is machine-verified against the logged runs by
`src/audit_numbers.py` (69 + 41 checks, 0 failures).

## Questions for authors

**Q1 — Is the calibration table (isotonic/Platt/Beta/Venn–Abers) based on the class-weighted
model only? Could the unweighted base also be reported?**
Addressed. Table II now has two blocks: the class-weighted base (uncalibrated, isotonic,
Platt, Beta, Venn–Abers) and the **unweighted base recommended for deployment**
(uncalibrated ECE 0.0038, mean predicted 0.139; isotonic 0.0034; Beta 0.0042). The text
states explicitly that the unweighted model "needs no recalibration step", so the
recommendation and the calibration evidence now come from the same model. The
weighted-vs-unweighted discrimination and net-benefit comparison (AUC 0.8265 vs 0.8267;
AUPRC 0.4229 vs 0.4234; net benefit 0.0785 vs 0.0787; specificity 0.623 vs 0.637) is
reported in the surrounding paragraph.

**Q2 — The SHAP bar chart lists features such as "ever_married" and "work_type", which are
not in the BRFSS Diabetes Health Indicators release.**
No change required — verified, not a defect in our figures. Every interpretability plot in
this repository was checked against the 21-feature BRFSS schema: `assets/figures/shap_top_features.png`
and `results/figures/shap_summary.png` plot only GenHlth, HighBP, Age, BMI, HighChol,
Income, Sex, CholCheck, MentHlth, HvyAlcoholConsump, HeartDiseaseorAttack, DiffWalk,
Education. The same check was run on `results/figures/feature_selectors.png`,
`age_gap_support.png`, `fairness_by_age.png`, `leakage_sensitivity.png`, `wave_drift.png`
and `ablation.png`. No figure in the release contains `ever_married` or `work_type`; the
reviewer appears to have inspected a different paper's figure. The manuscript's SHAP
section lists the same 14 features with their mean absolute SHAP values, so the text and
the figure can be cross-checked by the reader.

**Q3 — Why does the grouped held-out protocol (P3) score higher than the plain split
while the deduplicated test (P2) scores lower?**
Addressed in the leakage-audit paragraph. The ordering follows from what each protocol
evaluates: the plain split scores 50,736 rows once; the grouped protocol scores all
253,680 rows once, out of fold, with no covariate vector crossing a fold boundary, so its
evaluation set is 5.8× larger and the estimate correspondingly less noisy — a more
complete evaluation, not a better model; the deduplicated protocol keeps the plain
split's training set but restricts the test set to the 43,900 rows with no covariate twin
in training, which is the deployment-representative and most conservative case. We
therefore treat 0.8177 as the leakage-free estimate and report 0.8269 only with its
protocol attached.

**Q4 — How sensitive are the fairness gaps to the operating threshold? Can fairness be
summarized at the recommended sensitivity-targeted threshold rather than 0.5?**
Addressed. The evaluation-framework subsection states the threshold convention explicitly
(0.5 on raw scores for the subgroup table; recalibrated scores at a sensitivity-targeted
threshold for all operating-point results). Table IV reports the sex, income and education
TPR gaps **at the deployed operating point** (2015 in sample, and 2021 out of time) for
every threshold policy, and the surrounding text tracks the global-threshold gaps across
all three later waves (age 0.5111 → 0.7727, 0.7901, 0.7646; sex 0.0171 → 0.0201, 0.0229,
0.0275).

**Q5 — Can a survey-aware evaluation be reported (approximate BRFSS weights or a design-effect
sensitivity analysis)?**
Partially addressed, with the limitation stated. The release carries no design variables,
so we cannot reweight; we instead report the **design-effect sensitivity** of every
bootstrap interval (the widest TPR-gap interval grows from 0.235 at DEFF = 1 to 0.407 at
DEFF = 3, so conclusions are unchanged but our intervals are optimistic), and we add the
magnitude argument: the CDC's own analysis of aggregated state weights documents a
national-distribution error of 0.5 percentage points on a single demographic margin with
downstream distortion of diabetes prevalence estimates by group. The design-aware route
(weights, strata, PSUs) requires the raw BRFSS XPT files and a re-derivation of the
respondent-level merge; it is listed as the first item of future work and as a
prerequisite for any policy-facing use.

**Q6 — How would the label-noise refit be operationalized prospectively, and what are the
risks of overfitting to unresolvable artifacts?**
Addressed. The label-noise paragraph now states that the refit is a diagnostic, not a
proposed headline and not deployable practice: it is trained and tested on a different,
smaller sample (n = 248,462) whose train and test sets are themselves free of contradicting
pairs, so its AUC (0.8397) is not comparable with the reported protocols and is not
reported as our performance estimate. It would be optimistic for any deployed model,
because a respondent whose covariates match another's but whose outcome differs cannot be
labelled correctly at all. The contradicting rows actively degrade the fitted model — no
deterministic function of the covariates can satisfy them — so the headline AUC of 0.8265
is conservative with respect to label noise.

**Q7 — Were portable fairness post-processing schemes (equalized-odds post-processing with
per-group calibration, drift-aware thresholding) considered?**
Addressed. Per-group calibration before thresholding is reported as an alternative
stabiliser (sex gap 0.0171 → 0.0072; age gap 0.5111 → 0.4780; other attributes unchanged
or worse; fitted on a single split). The discussion states that portable post-processing
remains untested: equalized-odds post-processing fitted on one wave would inherit the
non-transferability we documented for per-attribute thresholds, and drift-aware variants
would require the design-based inference we lack. The failure of the thresholds is
diagnosed rather than asserted: per-level thresholds drift by at most 0.040 between 2015 and
2021, while within-level prevalence shifts by up to +0.029 (income level 8) and −0.059
(education level 2), so the sensitivity each level actually receives drifts between 0.772
and 0.895 instead of the targeted 0.85.

## Stated weaknesses

| Weakness | Status |
|---|---|
| No survey design variables (weights, PSUs, strata) | Acknowledged; design-effect sensitivity reported; full design-aware inference listed as future work and as a prerequisite for population claims (see Q5) |
| Label-noise refit may induce optimistic performance | Explicitly bounded as a diagnostic, not a headline (see Q6) |
| No cross-survey external validation | Stated as a genuine gap, with the published 0.794 → 0.717 shift cited; NHANES cross-validation listed as future work |
| Protocol AUC movements not reconciled | Reconciled in the leakage-audit paragraph (see Q3) |
| Other portable mitigations not evaluated | Per-group calibration reported; equalized-odds and drift-aware variants discussed with expected trade-offs (see Q7) |
| Formatting artifacts / mismatched SHAP figure | Verified false positive (see Q2); all figures cross-checked against the 21-feature schema |
| Threshold conventions vary across tables | Conventions stated once in the evaluation-framework subsection and referenced from every table caption |
| Survey-aware ML frameworks (SaML) / design-based learning not engaged | Cited and positioned in the limitations (design-aware modelling and design-based inference as the established remedy); the reviewer's own identifiers (2605.08963, 2403.19752) are covered by our design-based-inference references |
| Published near-0.99 AUCs need a replication attempt or methodological critique | Addressed with a quantitative critique: on the same 253,680-row dataset, duplicate profiles and fold construction move AUC by at most 0.010 across our three protocols, so the published figures are unlikely to be reproducible |

## Change log since round 6

- Added the unweighted-base block to the calibration table and the unweighted-calibration
  sentences to the text.
- Added an explicit reconciliation of the three held-out protocols (P1/P2/P3).
- Added the threshold-convention statement to the evaluation-framework subsection and
  referenced it from the table captions.
- Added the portable-post-processing trade-off discussion to the deployment paragraph.
- Bounded the label-noise refit explicitly and stated why it cannot be operationalized.
- Tightened the manuscript to the 6-page IEEE limit (from a 9-page first IEEEtran build):
  4 tables (Table II calibration, Table III fairness supports, Table IV threshold policies,
  Table I performance), 1 figure (SHAP), 15 references, ~5,000 words. All 110
  machine-checked numbers were preserved.
