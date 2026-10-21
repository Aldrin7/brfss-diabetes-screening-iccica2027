import json, os, re
ROOT = r"D:\Research\On-Progress\PIET Conference"
LOGS = os.path.join(ROOT, "results", "logs")
tex = open(os.path.join(ROOT, "manuscript.tex"), encoding="utf-8").read()

def g(n):
    p = os.path.join(LOGS, n)
    return json.load(open(p)) if os.path.exists(p) else {}

r1, nov = g("results.json"), g("novel_results.json")
a1, a2, a3, a4 = g("audit_results.json"), g("audit_results2.json"), g("audit_results3.json"), g("audit_results4.json")
a5, a6, a7, a8, a9 = (g("audit_results%d.json" % i) for i in (5, 6, 7, 8, 9))

checks = []
def chk(label, needle, ok):
    checks.append((label, needle, needle in tex, ok))

# performance
agg = r1["aggregate"]
chk("LR acc", "0.7308", abs(agg["LogisticRegression"]["accuracy_mean"]-0.7308)<5e-5)
chk("LR auc", "0.8212", abs(agg["LogisticRegression"]["auc_mean"]-0.8212)<5e-5)
chk("RF auc", "0.8261", abs(agg["RandomForest"]["auc_mean"]-0.8261)<5e-5)
chk("HGB auc", "0.8288", abs(agg["HistGradientBoosting"]["auc_mean"]-0.8288)<5e-5)
chk("HGB f1", "0.4426", abs(agg["HistGradientBoosting"]["f1_mean"]-0.4426)<5e-5)
# leakage
dup = a5["duplicates_consolidated"]
chk("dup all-cols 24206", "24{,}206", dup["dup_all_cols_incl_target"]==24206)
chk("dup feature 25772", "25{,}772", dup["dup_feature_vector_only"]==25772)
chk("conflicting 1963", "1{,}963", dup["dup_feature_vector_conflicting_label"]==1963)
chk("unique vectors 227908", "227{,}908", dup["n_unique_feature_vectors"]==227908)
lp = a5["leakage_protocols"]
chk("P1 HGB auc", "0.8269", abs(lp["P1_plain_stratified_80_20"]["HistGradientBoosting"]["roc_auc"]-0.8269)<5e-5)
chk("P2 HGB auc", "0.8177", abs(lp["P2_deduplicated_test"]["HistGradientBoosting"]["roc_auc"]-0.8177)<5e-5)
chk("P3 HGB auc", "0.8301", abs(lp["P3_grouped_heldout_HGB"]["HistGradientBoosting"]["roc_auc"]-0.8301)<5e-5)
chk("P2 n 43900", "43{,}900", lp["P2_deduplicated_test"]["HistGradientBoosting"]["n_test"]==43900)
# calibration
c = a5["calibrators"]
chk("uncal ece", "0.2367", abs(c["uncalibrated"]["ece"]-0.2367)<5e-5)
chk("iso ece", "0.0031", abs(c["isotonic"]["ece"]-0.0031)<5e-5)
chk("beta ece", "0.0026", abs(c["beta"]["ece"]-0.0026)<5e-5)
chk("beta brier", "0.0974", abs(c["beta"]["brier"]-0.0974)<5e-5)
chk("platt ece", "0.0083", abs(c["platt"]["ece"]-0.0083)<5e-5)
chk("va ece", "0.1534", abs(c["venn_abers"]["ece"]-0.1534)<5e-5)
# fairness
f = a1["fairness_diagnostics"]
chk("sex tpr gap", "0.0241", abs(f["Sex"]["fairness"]["tpr_gap"]-0.0241)<5e-5)
chk("age tpr gap", "0.7889", abs(f["Age"]["fairness"]["tpr_gap"]-0.7889)<5e-5)
chk("age min npos 14", "14 positives", f["Age"]["fairness"]["min_group_n_pos"]==14)
chk("income tpr gap", "0.2849", abs(f["Income"]["fairness"]["tpr_gap"]-0.2849)<5e-5)
# age robustness
rb = a3["age_robustness"]["by_min_support"]
chk("age gap all", "0.7879", abs(rb["1"]["tpr_gap"]-0.7879)<5e-5)
chk("age gap 500", "0.1436", abs(rb["500"]["tpr_gap"]-0.1436)<5e-5)
chk("ci width 0.340", "0.340", abs(rb["1"]["widest_group_tpr_ci"]-0.340)<1e-3)
chk("ci width 0.072", "0.072", abs(rb["500"]["widest_group_tpr_ci"]-0.072)<1e-3)
# per-wave
pw = a6["per_wave_calibration"]
chk("2017 transport ece", "0.0064", abs(pw["2017"]["isotonic_transported"]["ece"]-0.0064)<5e-5)
chk("2019 transport ece", "0.0062", abs(pw["2019"]["isotonic_transported"]["ece"]-0.0062)<5e-5)
chk("2021 transport ece", "0.0097", abs(pw["2021"]["isotonic_transported"]["ece"]-0.0097)<5e-5)
wd = a2["wave_drift"]
chk("2017 auc", "0.8338", abs(wd["2017"]["roc_auc"]-0.8338)<5e-5)
chk("2019 auc", "0.8256", abs(wd["2019"]["roc_auc"]-0.8256)<5e-5)
chk("2021 auc", "0.8198", abs(wd["2021"]["roc_auc"]-0.8198)<5e-5)
chk("thr 2017", "0.420", abs(wd["2017"]["thr_for_sens085"]-0.420)<5e-4)
chk("thr 2021", "0.395", abs(wd["2021"]["thr_for_sens085"]-0.395)<5e-4)
# transport penalty
tp = a9["per_wave_transport_vs_refit"]
chk("transport penalty 0.0035", "0.0035", abs(tp["2017"]["transport_penalty_ece"]-0.0035)<5e-5)
# group-specific thresholds
gs = a7["group_specific_thresholds"]["0.85"]
chk("gs sex gap", "0.0014", abs(gs["group_specific"]["Sex"]["tpr_gap"]-0.0014)<5e-5)
chk("gs edu gap", "0.1699", abs(gs["group_specific"]["Education"]["tpr_gap"]-0.1699)<5e-5)
chk("gs income gap", "0.2300", abs(gs["group_specific"]["Income"]["tpr_gap"]-0.2300)<5e-5)
chk("gs sens", "0.857", abs(gs["group_specific"]["overall_sensitivity"]-0.857)<5e-4)
chk("gs spec", "0.637", abs(gs["group_specific"]["overall_specificity"]-0.637)<5e-4)
# constrained
ct = a8["constrained_threshold_nondegenerate"]
chk("delta 0.30 thr", "0.140", abs(ct["s0.85_d0.30"]["threshold"]-0.140)<6e-3)
chk("delta 0.10 thr", "0.045", abs(ct["s0.85_d0.10"]["threshold"]-0.045)<6e-3)
chk("delta 0.50 thr", "0.400", abs(ct["s0.85_d0.50"]["threshold"]-0.400)<6e-3)
chk("baseline sens", "0.799", abs(ct["s0.85_d0.50"]["sens"]-0.867)>0 and abs(a8["constrained_threshold_baseline"]["sens"]-0.799)<5e-4)
# shap stability
ss = a5["shap_stability"]
chk("jaccard seeds", "0.964", abs(ss["mean_jaccard_across_seeds"]-0.964)<5e-4)
chk("jaccard waves", "0.818", abs(ss["mean_jaccard_2015_vs_waves"]-0.818)<5e-4)
# severity
sv = a8["severity_stratified"]
chk("gh12 auc", "0.8145", abs(sv["GenHlth 1-2 (good/very good)"]["within_stratum_auc"]-0.8145)<5e-5)
chk("gh45 auc", "0.7199", abs(sv["GenHlth 4-5 (poor/very poor)"]["within_stratum_auc"]-0.7199)<5e-5)
chk("bmi<30 auc", "0.8263", abs(sv["BMI < 30"]["within_stratum_auc"]-0.8263)<5e-5)
# dca
ds = a5["decision_curve"]["selected"]
chk("dca 0.10 nb", "0.068", abs(ds["0.10"]["model"]-0.068303)<1e-3)
chk("dca 0.20 nb", "0.009", abs(ds["0.20"]["model"]-0.009402)<1e-3)
# auprc
ap = a5["auprc"]
chk("hgb auprc", "0.4229", abs(ap["HistGradientBoosting"]["auprc"]-0.4229)<5e-5)
chk("lr auprc", "0.3926", abs(ap["LogisticRegression"]["auprc"]-0.3926)<5e-5)
# design effect
de = a5["design_effect_sensitivity"]["by_deff"]
chk("deff3 width", "0.407", abs(de["3.0"]["widest_group"]["ci_width"]-0.4074)<1e-3)
# scenario
sc = a4["scenario_analysis"]
chk("bmi5 raw", "0.0694", abs(sc["BMI +5 (1 SD approx)"]["mean_prob_change"]-0.0694)<1e-3)
chk("bmi5 cal", "0.0410", abs(a7["scenario_subgroup"]["BMI +5"]["calibrated"]["mean_change"]-0.0410)<1e-3)
# ablation / mq
ab = r1["ablation"]; mq = nov["minimal_questionnaire"]
chk("abl demo", "0.7104", abs(ab["demographic_only"]["roc_auc"]-0.7104)<5e-5)
chk("abl clin", "0.8241", abs(ab["plus_clinical"]["roc_auc"]-0.8241)<5e-5)
chk("abl life", "0.7475", abs(ab["plus_lifestyle"]["roc_auc"]-0.7475)<5e-5)
chk("mq k5", "0.8186", abs(mq["5"]["roc_auc"]-0.8186)<5e-5)
chk("mq k11", "0.8257", abs(mq["11"]["roc_auc"]-0.8257)<5e-5)
fs = a2["feature_selectors"]
chk("mi k10", "0.8213", abs(fs["mutual_info_k10"]["auc"]-0.8213)<5e-5)
chk("corr k10", "0.8218", abs(fs["correlation_k10"]["auc"]-0.8218)<5e-5)
chk("rand k10", "0.7784", abs(fs["random_k10"]["auc_mean"]-0.7784)<5e-5)
# paired auc
pa = a4["auc_pairwise"]["pairs"]
chk("hgb-rf dAUC", "0.0034", abs(pa["RandomForest vs HistGradientBoosting"]["delta_auc"]*-1-0.0034)<5e-5)
chk("hgb-lr dAUC", "0.0073", abs(pa["LogisticRegression vs HistGradientBoosting"]["delta_auc"]*-1-0.0073)<5e-5)
# hp search
hs = a2["hyperparameter_search"]
chk("hp hgb", "0.8303", abs(hs["HistGradientBoosting"]["best_cv_auc"]-0.8303)<5e-5)
chk("hp rf", "0.8275", abs(hs["RandomForest"]["best_cv_auc"]-0.8275)<5e-5)
chk("hp lr", "0.8233", abs(hs["LogisticRegression"]["best_cv_auc"]-0.8233)<5e-5)

fails = [c for c in checks if not (c[2] and c[3])]
print("checks: %d | failures: %d" % (len(checks), len(fails)))
for label, needle, in_tex, ok in fails:
    print("  FAIL %-22s needle=%-12s in_tex=%s value_ok=%s" % (label, needle, in_tex, ok))
