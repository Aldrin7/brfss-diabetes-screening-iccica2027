# Review responses — round 8 (internal record)

Paper: "What the Audits Change: An Interpretable and Fairness-Aware Screening Framework for Diabetes from Population Health Surveys"
Reviewer: PaperReview.ai round 8, submitted 2026-10-21 15:48 IST, reviewed 16:15 IST
Outcome: **"I recommend acceptance after minor revisions"** — triple scores 5x+1 (claims support,
experimental soundness, writing clarity, question importance, value to community), 0 on
prior-work context and originality.

Round 8 was run deliberately as a **verification pass on the exact 6-page submission
artifact** (the previous rounds reviewed a 9-page manuscript before the compression), under
two rules: fix errors only, and add no content that costs space. Both rules were applied:
the page count is unchanged at 6, and every added statement is backed by a logged run.

New computation: `src/audit_round8.py` -> `results/logs/audit_results13.json`
(64/16/20 split, HGB `max_iter=150, lr=0.1, leaves=31, class_weight=balanced, seed=42`,
isotonic fitted on the 2015 calibration split, recommended threshold 0.115). All 11 new
numbers are machine-checked by `src/audit_numbers.py` (now 69 + 41 + 11 = 121 checks, 0 failures).

## Questions for authors

**Q1 — Clarify the Venn–Abers variant and prior-correction details; provide the fitted
calibration curves or parameters; could the poor performance be replicated on an unweighted
base model?**
Addressed in the calibration subsection. The paper now states the objective and the two
constraints of the implementation (isotonic regression of the outcome on the base score plus a
self-consistency correction reconciling predicted and observed positives in the calibration
split), explains that the correction assumes mildly and uniformly over-confident base scores
and that a class-weighted ensemble with mean predicted probability $2.7$ times the true
prevalence violates that assumption, and adds the unweighted-base comparison: on the
recommended unweighted model the calibrator is unnecessary because those scores are already
calibrated (ECE $0.0038$), which is the completeness check the reviewer asked for. Fitted
curves and parameters are in the released logs (`results/logs/audit_results5.json`).

**Q2 — In Table IV the sex-specific policy shows the same in-sample sex TPR gap as the global
threshold; is that a rounding artifact? Could you add per-level sensitivities for each policy?**
Addressed in the operating-point subsection: the sex policy reproduces the global row because
the sex gap is already the smallest of the four attributes, not through a rounding artifact.
Per-policy per-level sensitivity distributions are the mechanism we already document
(per-level sensitivity drifts between $0.772$ and $0.895$ rather than the targeted $0.85$),
and the underlying per-level values are in `results/logs/audit_results11.json`
(`group_thresholds_out_of_time`); the full per-policy/per-level matrix is logged rather than
tabulated, to keep the paper within the six-page limit.

**Q3 — For the contradictory-label duplicates, are these truly duplicated respondents or
distinct individuals with identical covariates? Any sensitivity with a near-identical
definition?**
Addressed, and this is the most important change in the paper. The leakage subsection now
reads: the 1,963 contradicting-label rows "reflect either duplicated records with
inconsistent labels or distinct respondents who share every measured covariate yet differ in
outcome, so the count bounds label inconsistency rather than measuring it." This qualifies the
original phrasing ("a direct measure of label inconsistency"), which overstated what the
count can show, and it is consistent with the label-noise subsection, which already argues
that no deterministic function of the covariates can satisfy the contradicting rows. The
label-noise refit (AUC $0.8265 \rightarrow 0.8397$ on a different, smaller sample) remains a
diagnostic and is still not reported as a performance estimate.

**Q4 — Report subgroup AUCs and per-group calibration (ECE/Brier) for 2015 and 2021, especially
for the age groups.**
Addressed with new computation. Over the seven support-retained age bands, band-level AUC
ranges from $0.7030$ to $0.8245$ in 2015 and $0.7065$ to $0.8064$ in 2021, with per-band ECE
at most $0.0208$ and $0.0246$ respectively. The paper now states this and draws the inference
the reviewer was after: the age TPR gap is **not** a calibration failure, it is a
discrimination/scale phenomenon across bands. Per-band n, n_pos, prevalence, AUC, ECE, Brier,
TPR, FPR and net benefit for 2015 and 2021 are in `results/logs/audit_results13.json`.

**Q5 — Exact age-band definitions and positive counts per retained band.**
Documented. The 13 bands are the standard BRFSS age categories (18-24, 25-29, ..., 80+); the
full per-band n and positive counts on 2015, with the retention flag at the 500-positive
criterion, are released as `results/logs/age_band_supports.csv`. On the 2015 test set the
seven retained bands are 50-54 (605 positives), 55-59 (812), 60-64 (1192), 65-69 (1319),
70-74 (1002), 75-79 (706) and 80+ (643), which reproduces the "seven bands" claim already in
the paper. The table itself is logged rather than printed, to preserve page space.

**Q6 — Did you evaluate per-group recalibration (group-wise isotonic) before applying a single
transported global threshold? How did per-group calibration transport compare to overall
calibration transport?**
Addressed with new computation and it strengthens the paper's recommendation. In sample,
per-group calibration before thresholding narrows the sex gap ($0.0171 \rightarrow 0.0072$) and
the age gap modestly ($0.5111 \rightarrow 0.4780$) but leaves other attributes unchanged or
worse. Transported to 2021, the same age-wise recalibration gives an overall ECE of $0.0116$
against $0.0097$ for the single global map, i.e. per-group recalibration does **not** transport
better than one global calibrator. This is now stated in the paper.

**Q7 — Net benefit stratified by key subgroups at the recommended operating point.**
Addressed with new computation. Net benefit at the recommended threshold (threshold
probability 0.10) is positive in every retained age band from 50-54 upward in 2021, ranging
$0.0692$--$0.1556$. The paper now reports the range; the full per-band table is in
`results/logs/audit_results13.json`.

**Q8 — Did you evaluate focal loss or asymmetric loss as an alternative to class weights?**
Answered honestly in the paper: cost-sensitive loss functions were not evaluated, so that
alternative remains open. The paper's recommendation does not depend on them: it is to train
unweighted and re-weight at the operating point, which is orthogonal to the loss function.

## Stated weaknesses

| Weakness | Status |
|---|---|
| No survey design variables (weights, PSUs, strata) | Acknowledged; design-effect sensitivity reported; full design-aware inference stated as the prerequisite for population claims |
| Race/ethnicity missing | Acknowledged; fairness limited to sex, age, income, education |
| **Contradicting-label duplicates may be distinct individuals, not label error** | **Fixed** — the count now bounds label inconsistency rather than measuring it (see Q3) |
| No cross-survey external validation (e.g. NHANES) | Acknowledged as a genuine gap; the published $0.794 \rightarrow 0.717$ transfer drop is cited |
| Venn–Abers only qualitatively explained | Fixed — objective and constraints stated, unweighted-base comparison added (see Q1) |
| Modest hyperparameter-search budget | Acknowledged in the limitations; ordering is preserved under the randomized search, so the central claims do not depend on it |
| Table IV sex-specific row reads as redundant | Clarified in the text (see Q2) |
| Raw vs calibrated scores mixed across analyses | Conventions stated once in the evaluation subsection and referenced from every table caption; the new per-band results are all computed on recalibrated scores at the single recommended threshold |
| Survey-aware ML guideline literature (SaML) not cited | **Fixed** — SaML \cite{saml} added and connected to the design-aware requirement |
| No comparison to multi-task / cross-domain screening frameworks | **Fixed** — CardioMeta \cite{cardiometa} added as a contrast: multi-task frameworks that pair leakage control with calibration reach comparable macro-AUROC on similar survey data, and this single-task audit complements them by isolating the pitfalls |

## What was deliberately not changed

- **The 6-page limit.** No new table or figure was added; the new results are reported as
  compact sentences and the tables are released as logs. The paper is still 6 pages, so no
  over-length charge is incurred.
- **Reported numbers.** No logged value was altered. The 41 + 69 + 11 machine checks all pass.
- **The recommendations.** Unweighted training, a single global sensitivity-targeted threshold,
  and no subgroup thresholds: the round-8 evidence (per-group recalibration transporting worse
  than the global map) supports them rather than contradicting them.

## Change log since round 8

1. Leakage subsection: contradicting-label caveat (Q3, weakness 3).
2. Fairness subsection: per-band AUC, ECE and net benefit for 2015 and 2021 over the seven
   support-retained bands (Q4, Q7).
3. Operating-point subsection: why the sex-specific policy reproduces the global row (Q2);
   per-group calibration transport to 2021 (ECE 0.0116 vs 0.0097) (Q6).
4. Calibration subsection: Venn–Abers objective and constraints, plus the unweighted-base
   comparison (Q1).
5. Class-weight subsection: cost-sensitive loss functions not evaluated (Q8).
6. Discussion: tightened positioning paragraph; added the SaML guideline connection and the
   multi-task (CardioMeta) contrast; removed the duplicated calibrator-ranking discussion.
7. New: `src/audit_round8.py`, `results/logs/audit_results13.json`,
   `results/logs/age_band_supports.csv`, 11 new machine checks in `src/audit_numbers.py`.
8. New references (both identifiers verified on arXiv): SaML `arXiv:2605.08963` (Oh, Zheng,
   Feng, Bui), CardioMeta `arXiv:2607.15721` (Hossain, Shayoni, Mridha, Shin).
