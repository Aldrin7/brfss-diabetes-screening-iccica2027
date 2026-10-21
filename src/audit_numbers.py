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
chk("gs sex gap raw", "0.0014", abs(gs["group_specific"]["Sex"]["tpr_gap"]-0.0014)<5e-5)
chk("gs sens raw", "0.857", abs(gs["group_specific"]["overall_sensitivity"]-0.857)<5e-4)
chk("gs spec raw", "0.637", abs(gs["group_specific"]["overall_specificity"]-0.637)<5e-4)
ct = a8["constrained_threshold_nondegenerate"]
chk("delta 0.30 thr", "0.140", abs(ct["s0.85_d0.30"]["threshold"]-0.140)<6e-3)
chk("delta 0.10 thr", "0.045", abs(ct["s0.85_d0.10"]["threshold"]-0.045)<6e-3)
chk("delta 0.50 thr", "0.400", abs(ct["s0.85_d0.50"]["threshold"]-0.400)<6e-3)
chk("delta 0.30 spec", "0.344", abs(ct["s0.85_d0.30"]["spec"]-0.344)<5e-4)
chk("delta 0.10 spec", "0.147", abs(ct["s0.85_d0.10"]["spec"]-0.147)<5e-4)
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

# ---------- round-4 additions ----------
a10 = g("audit_results10.json"); a11 = g("audit_results11.json"); a12 = g("audit_results12.json")
print("\n=== round-4 additions ===")
checks2 = []
def chk2(label, needle, ok):
    checks2.append((label, needle, needle in tex, ok))

w = a10["weighted_vs_unweighted"]
chk2("cw auc", "0.8265", abs(w["class_weighted"]["roc_auc"]-0.8265)<5e-5)
chk2("unw auc", "0.8267", abs(w["unweighted"]["roc_auc"]-0.8267)<5e-5)
chk2("unw auprc", "0.4234", abs(w["unweighted"]["auprc"]-0.4234)<5e-5)
chk2("unw raw ece", "0.0038", abs(w["unweighted"]["raw"]["ece"]-0.0038)<5e-5)
chk2("cw mean pred", "0.376", abs(w["class_weighted"]["raw"]["mean_pred"]-0.376)<5e-4)
chk2("unw mean pred", "0.139", abs(w["unweighted"]["raw"]["mean_pred"]-0.139)<5e-4)
chk2("unw spec", "0.637", abs(w["unweighted"]["specificity"]-0.637)<5e-4)
chk2("unw nb", "0.0787", abs(w["unweighted"]["net_benefit"]-0.0787)<5e-5)
chk2("cw nb", "0.0785", abs(w["class_weighted"]["net_benefit"]-0.0785)<5e-5)
chk2("unw thr", "0.115", abs(w["unweighted"]["sens_targeted_threshold"]-0.115)<6e-4)
chk2("cw sens", "0.865", abs(w["class_weighted"]["sensitivity"]-0.865)<5e-4)
chk2("unw sens", "0.857", abs(w["unweighted"]["sensitivity"]-0.857)<5e-4)
mc = a10["multi_calibration"]
chk2("mc sex gap", "0.0072", abs(mc["per_group_calib_then_group_thr"]["Sex"]["tpr_gap"]-0.0072)<5e-5)
chk2("mc age gap", "0.4780", abs(mc["per_group_calib_then_group_thr"]["Age"]["tpr_gap"]-0.4780)<5e-5)
ln = a10["label_noise"]
chk2("ln clean auc", "0.8397", abs(ln["clean_auc"]-0.8397)<5e-5)
chk2("ln removed", "5{,}218", ln["n_rows_removed"]==5218)
chk2("ln age gap", "0.6671", abs(ln["clean_gaps"]["Age"]["tpr_gap"]-0.6671)<5e-5)
chk2("ln edu gap", "0.1793", abs(ln["clean_gaps"]["Education"]["tpr_gap"]-0.1793)<5e-5)
eb = a10["ece_binning_robustness"]
chk2("ece ew15", "0.0049", abs(eb["equal_width_15"]["isotonic"]-0.0049)<5e-5)
chk2("ece em10", "0.0044", abs(eb["equal_mass_10"]["isotonic"]-0.0044)<5e-5)
chk2("ece tw", "0.2155", abs(eb["threshold_weighted"]["uncalibrated"]-0.2155)<5e-5)
cf = a10["counterfactual_flips"]
chk2("cf bmi", "88.2", abs(cf["BMI"]["flip_rate"]-0.8820)<5e-4)
chk2("cf genhlth", "67.5", abs(cf["GenHlth"]["flip_rate"]-0.6745)<5e-4)
chk2("cf age", "48.0", abs(cf["Age"]["flip_rate"]-0.4800)<5e-4)
p15 = a11["policy_comparison_2015"]
chk2("pol age gap", "0.9087", abs(p15["Age"]["Age"]["tpr_gap"]-0.9087)<5e-5)
chk2("pol inc gap", "0.0899", abs(p15["Income"]["Income"]["tpr_gap"]-0.0899)<5e-5)
chk2("pol edu gap", "0.0933", abs(p15["Education"]["Education"]["tpr_gap"]-0.0933)<5e-5)
oot21 = a11["policy_out_of_time"]["2021"]
chk2("oot inc gap", "0.8647", abs(oot21["Income"]["Income"]["tpr_gap"]-0.8647)<5e-5)
chk2("oot edu gap", "0.7128", abs(oot21["Education"]["Education"]["tpr_gap"]-0.7128)<5e-5)
chk2("oot age gap 2017", "0.8937", abs(a11["policy_out_of_time"]["2017"]["Age"]["Age"]["tpr_gap"]-0.8937)<5e-5)
chk2("oot age gap 2019", "0.8992", abs(a11["policy_out_of_time"]["2019"]["Age"]["Age"]["tpr_gap"]-0.8992)<5e-5)
chk2("oot global sens", "0.842", abs(oot21["Age"]["overall_sensitivity"]-0.819)<5e-3 and abs(a12["global_sens_targeted"]["2021"]["transported_threshold"]["overall_sensitivity"]-0.842)<5e-4)
g15 = a12["global_sens_targeted"]["2015_test"]
chk2("g15 age gap", "0.5111", abs(g15["Age"]["tpr_gap"]-0.5111)<5e-5)
chk2("g15 sex gap", "0.0171", abs(g15["Sex"]["tpr_gap"]-0.0171)<5e-5)
chk2("g15 spec", "0.623", abs(g15["overall_specificity"]-0.623)<5e-4)
for yr, ag, sg in [("2017", "0.7727", "0.0201"), ("2019", "0.7901", "0.0229"), ("2021", "0.7646", "0.0275")]:
    e = a12["global_sens_targeted"][yr]["transported_threshold"]
    chk2("g%s age gap" % yr, ag, abs(e["Age"]["tpr_gap"]-float(ag))<5e-5)
    chk2("g%s sex gap" % yr, sg, abs(e["Sex"]["tpr_gap"]-float(sg))<5e-5)

fails2 = [c for c in checks2 if not (c[2] and c[3])]
print("round-4 checks: %d | failures: %d" % (len(checks2), len(fails2)))
for label, needle, in_tex, ok in fails2:
    print("  FAIL %-22s needle=%-12s in_tex=%s value_ok=%s" % (label, needle, in_tex, ok))
