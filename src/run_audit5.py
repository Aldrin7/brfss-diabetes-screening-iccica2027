import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             f1_score, confusion_matrix)
from sklearn.isotonic import IsotonicRegression

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results5.json")


def load_year(year):
    if year == 2015:
        p = os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv")
    else:
        p = os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % year)
    return pd.read_csv(p)


def hgb(seed=42, **kw):
    params = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                  class_weight="balanced", random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def ece_mce(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0; mce = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        conf = p[m].mean(); acc = y[m].mean()
        ece += (m.sum() / len(y)) * abs(acc - conf)
        mce = max(mce, abs(acc - conf))
    return float(ece), float(mce)


def cal_block(y, p):
    e, mc = ece_mce(y, p)
    return {"n": int(len(y)), "brier": float(brier_score_loss(y, p)), "ece": e, "mce": mc,
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


# ---------- calibrators ----------
class VennAbers:
    """Venn-Abers / AB-isotonic: isotonic on P(Y=1|p) with self-consistent prior correction."""
    def fit(self, p, y):
        p = np.asarray(p); y = np.asarray(y)
        n_pos = int((p > 0.5).sum()); n_neg = int((p <= 0.5).sum())
        self.prior = n_pos / max(n_pos + n_neg, 1)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        self.n_pos, self.n_neg = n_pos, n_neg
        return self

    def predict(self, p):
        q = self.iso.predict(np.asarray(p))
        return (self.n_pos * q + self.n_neg * self.prior) / max(self.n_pos + self.n_neg, 1)


class BetaCal:
    """Beta calibration (Kull et al. 2017): logistic on [logit(p), -log(1-p)]."""
    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(p / (1 - p)), -np.log(1 - p)])
        self.lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(p / (1 - p)), -np.log(1 - p)])
        return self.lr.predict_proba(X)[:, 1]


def make_models(seed=42):
    return {
        "LogisticRegression": Pipeline([("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))]),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=20,
            class_weight="balanced", random_state=seed, n_jobs=1),
        "HistGradientBoosting": hgb(seed),
    }


def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    y_all = df[TARGET].astype(int).values
    X_all = df[FEATURES]

    # ============ A. CONSOLIDATED LEAKAGE PROTOCOL ============
    print("=== A. Consolidated duplicate / leakage protocol ===", flush=True)
    dup_all = df.duplicated(keep="first")
    dup_feat = df.duplicated(subset=FEATURES, keep="first")
    # rows whose feature vector repeats an earlier row but whose label differs
    lab = df[TARGET].astype(int)
    first_lab = lab.groupby([df[c] for c in FEATURES], sort=False).transform("first")
    conflicting = int((dup_feat & (lab.values != first_lab.values)).sum())
    out["duplicates_consolidated"] = {
        "n_rows": int(len(df)),
        "dup_all_cols_incl_target": int(dup_all.sum()),
        "dup_feature_vector_only": int(dup_feat.sum()),
        "dup_feature_vector_conflicting_label": conflicting,
        "pct_feature_dup": float(dup_feat.mean()),
        "n_unique_feature_vectors": int(len(df.drop_duplicates(subset=FEATURES))),
    }
    print("  all-cols (features+target) duplicates: %d" % out["duplicates_consolidated"]["dup_all_cols_incl_target"], flush=True)
    print("  feature-vector duplicates: %d (%.2f%%), of which %d carry a conflicting label" % (
        out["duplicates_consolidated"]["dup_feature_vector_only"],
        100 * out["duplicates_consolidated"]["pct_feature_dup"], conflicting), flush=True)

    # three held-out protocols
    protos = {}
    # P1: plain stratified split (as reported)
    Xtr, Xte, ytr, yte = train_test_split(X_all, lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    p1 = {}
    for name, m in make_models(42).items():
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        p1[name] = {"roc_auc": float(roc_auc_score(yte, pr)),
                    "auprc": float(average_precision_score(yte, pr)),
                    "f1": float(f1_score(yte, (pr >= 0.5).astype(int), zero_division=0))}
    protos["P1_plain_stratified_80_20"] = p1
    print("  P1 plain: %s" % {k: round(v["roc_auc"], 4) for k, v in p1.items()}, flush=True)

    # P2: deduplicated test set (drop test rows whose feature vector appears in train)
    train_keys = set(map(tuple, Xtr[FEATURES].to_numpy()))
    te_mask = np.array([tuple(r) not in train_keys for r in Xte[FEATURES].to_numpy()])
    Xte2, yte2 = Xte[te_mask], yte[te_mask]
    p2 = {}
    for name, m in make_models(42).items():
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte2)[:, 1]
        p2[name] = {"roc_auc": float(roc_auc_score(yte2, pr)),
                    "auprc": float(average_precision_score(yte2, pr)),
                    "f1": float(f1_score(yte2, (pr >= 0.5).astype(int), zero_division=0)),
                    "n_test": int(len(yte2))}
    protos["P2_deduplicated_test"] = p2
    print("  P2 dedup-test (n=%d): %s" % (len(yte2), {k: round(v["roc_auc"], 4) for k, v in p2.items()}), flush=True)

    # P3: grouped held-out (identical feature vectors never cross the split)
    groups = pd.util.hash_pandas_object(X_all, index=False).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for tr_i, te_i in sgkf.split(X_all, y_all, groups):
        m = hgb(42); m.fit(X_all.iloc[tr_i], y_all[tr_i])
        oof[te_i] = m.predict_proba(X_all.iloc[te_i])[:, 1]
    protos["P3_grouped_heldout_HGB"] = {
        "HistGradientBoosting": {"roc_auc": float(roc_auc_score(y_all, oof)),
                                  "auprc": float(average_precision_score(y_all, oof)),
                                  "n_test": int(len(y_all))}}
    print("  P3 grouped held-out (HGB): AUC=%.4f AUPRC=%.4f" % (
        protos["P3_grouped_heldout_HGB"]["HistGradientBoosting"]["roc_auc"],
        protos["P3_grouped_heldout_HGB"]["HistGradientBoosting"]["auprc"]), flush=True)
    out["leakage_protocols"] = protos
    save_json(out, OUTPATH)

    # ============ C. ALTERNATIVE CALIBRATORS ============
    print("\n=== C. Alternative calibrators (2015 calib split) ===", flush=True)
    Xrest, Xte3, yrest, yte3 = train_test_split(X_all, lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr3, Xcal, ytr3, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr3 = Xtr3.reset_index(drop=True); ytr3 = ytr3.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte3 = Xte3.reset_index(drop=True); yv3 = yte3.reset_index(drop=True).values
    m = hgb(42); m.fit(Xtr3, ytr3)
    p_cal = m.predict_proba(Xcal)[:, 1]
    p_te = m.predict_proba(Xte3)[:, 1]
    cals = {}
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    cals["isotonic"] = (iso.predict(p_te),)
    pl = LogisticRegression().fit(p_cal.reshape(-1, 1), ycal.values)
    cals["platt"] = (pl.predict_proba(p_te.reshape(-1, 1))[:, 1],)
    bt = BetaCal().fit(p_cal, ycal.values)
    cals["beta"] = (bt.predict(p_te),)
    va = VennAbers().fit(p_cal, ycal.values)
    cals["venn_abers"] = (va.predict(p_te),)
    cal_res = {"uncalibrated": cal_block(yv3, p_te)}
    for k, (pv,) in cals.items():
        cal_res[k] = cal_block(yv3, pv)
        print("  %-13s brier=%.4f ECE=%.4f MCE=%.4f mean_pred=%.3f" % (
            k, cal_res[k]["brier"], cal_res[k]["ece"], cal_res[k]["mce"], cal_res[k]["mean_pred"]), flush=True)
    out["calibrators"] = cal_res
    save_json(out, OUTPATH)

    # ============ D. AUPRC + DECISION CURVE (net benefit) ============
    print("\n=== D. AUPRC and decision-curve net benefit ===", flush=True)
    prevalence = float(yv3.mean())
    dca = {"prevalence": prevalence, "thresholds": [], "model": [], "treat_all": [], "treat_none": []}
    for t in np.arange(0.05, 0.55, 0.01):
        pred = p_te >= t
        tp = float(((pred) & (yv3 == 1)).sum()); fp = float(((pred) & (yv3 == 0)).sum())
        n = len(yv3)
        nb = tp / n - (fp / n) * (t / (1 - t))
        dca["thresholds"].append(round(float(t), 3))
        dca["model"].append(round(nb, 6))
        dca["treat_all"].append(round(prevalence - (1 - prevalence) * (t / (1 - t)), 6))
        dca["treat_none"].append(0.0)
    # net benefit at selected thresholds
    sel = {}
    for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
        i = min(range(len(dca["thresholds"])), key=lambda k: abs(dca["thresholds"][k] - t))
        sel["%.2f" % t] = {"model": dca["model"][i], "treat_all": dca["treat_all"][i],
                           "treat_none": 0.0, "threshold": dca["thresholds"][i]}
    dca["selected"] = sel
    out["decision_curve"] = dca
    au = {}
    for name, mm in make_models(42).items():
        mm.fit(Xtr3, ytr3)
        pr = mm.predict_proba(Xte3)[:, 1]
        au[name] = {"auprc": float(average_precision_score(yv3, pr)),
                    "roc_auc": float(roc_auc_score(yv3, pr))}
        print("  %-22s AUPRC=%.4f (ROC-AUC=%.4f)" % (name, au[name]["auprc"], au[name]["roc_auc"]), flush=True)
    out["auprc"] = au
    save_json(out, OUTPATH)

    # ============ E. SHAP STABILITY (Jaccard across seeds and waves) ============
    print("\n=== E. SHAP top-10 stability (Jaccard) ===", flush=True)
    import shap
    ranks = {}
    for seed in [42, 7, 123, 2024, 99]:
        mm = hgb(seed); mm.fit(Xtr3, ytr3)
        sv = shap.TreeExplainer(mm).shap_values(Xte3.sample(2000, random_state=42))
        if isinstance(sv, list): sv = sv[1]
        order = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
        ranks["seed_%d" % seed] = order[:10]
    m42 = hgb(42); m42.fit(Xtr3, ytr3)
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr)
        sv = shap.TreeExplainer(m42).shap_values(wy[FEATURES].sample(2000, random_state=42))
        if isinstance(sv, list): sv = sv[1]
        order = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
        ranks["wave_%d" % yr] = order[:10]
    def jac(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)
    keys = list(ranks.keys())
    seeds_only = [k for k in keys if k.startswith("seed_")]
    wave_only = [k for k in keys if k.startswith("wave_")]
    j_seed = [jac(ranks["seed_42"], ranks[k]) for k in seeds_only]
    j_wave = [jac(ranks["seed_42"], ranks[k]) for k in wave_only]
    out["shap_stability"] = {
        "rankings": ranks,
        "mean_jaccard_across_seeds": float(np.mean(j_seed)),
        "min_jaccard_across_seeds": float(np.min(j_seed)),
        "mean_jaccard_2015_vs_waves": float(np.mean(j_wave)),
        "min_jaccard_2015_vs_waves": float(np.min(j_wave)),
    }
    print("  Jaccard across seeds: mean=%.3f min=%.3f" % (
        out["shap_stability"]["mean_jaccard_across_seeds"], out["shap_stability"]["min_jaccard_across_seeds"]), flush=True)
    print("  Jaccard 2015 vs later waves: mean=%.3f min=%.3f" % (
        out["shap_stability"]["mean_jaccard_2015_vs_waves"], out["shap_stability"]["min_jaccard_2015_vs_waves"]), flush=True)
    save_json(out, OUTPATH)

    # ============ F. CONSTRAINED THRESHOLD OPTIMIZATION ============
    print("\n=== F. Constrained threshold optimization (min FN s.t. bounded age TPR gap) ===", flush=True)
    ages = Xte3["Age"].values
    grid = []
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p_te >= t).astype(int)
        fn = int(((pred == 0) & (yv3 == 1)).sum())
        per = {}
        for a in sorted(set(ages)):
            msk = ages == a
            npos = int((yv3[msk] == 1).sum())
            if npos < 20: continue
            per[a] = float(((pred[msk] == 1) & (yv3[msk] == 1)).sum() / npos)
        gap = (max(per.values()) - min(per.values())) if per else float("nan")
        grid.append({"t": float(t), "fn": fn, "age_tpr_gap": float(gap)})
    cons = {}
    for delta in [0.05, 0.10, 0.20, 0.30, 0.50, 1.01]:
        feas = [g for g in grid if g["age_tpr_gap"] <= delta]
        if feas:
            best = min(feas, key=lambda g: g["fn"])
            cons["%.2f" % delta] = {"feasible": True, "threshold": best["t"], "fn": best["fn"],
                                    "achieved_gap": best["age_tpr_gap"],
                                    "fn_rate": best["fn"] / max(int((yv3 == 1).sum()), 1)}
        else:
            cons["%.2f" % delta] = {"feasible": False}
        print("  max age TPR gap <= %.2f : %s" % (delta, cons["%.2f" % delta]), flush=True)
    out["constrained_threshold"] = {"grid": grid, "by_delta": cons}
    save_json(out, OUTPATH)

    # ============ G. DESIGN-EFFECT SENSITIVITY ============
    print("\n=== G. Design-effect sensitivity for subgroup CIs ===", flush=True)
    prev = json.load(open(os.path.join(LOGS, "audit_results.json")))
    fair = prev.get("fairness_diagnostics", {})
    deff = {}
    for d in [1.0, 1.5, 2.0, 3.0]:
        mult = float(np.sqrt(d))
        entry = {"multiplier": mult, "groups": {}}
        for g, v in fair.items():
            w = v["bootstrap"]["tpr_gap_ci95"][1] - v["bootstrap"]["tpr_gap_ci95"][0]
            entry["groups"][g] = {"tpr_gap_ci_width": float(w * mult),
                                  "tpr_gap_ci95_design_scaled": [v["bootstrap"]["tpr_gap_ci95"][0] * mult,
                                                                v["bootstrap"]["tpr_gap_ci95"][1] * mult]}
        worst = max(entry["groups"].items(), key=lambda kv: kv[1]["tpr_gap_ci_width"])
        entry["widest_group"] = {"stratifier": worst[0], "ci_width": worst[1]["tpr_gap_ci_width"]}
        deff[str(d)] = entry
        print("  DEFF=%.1f -> widest TPR-gap CI width = %.4f (%s)" % (
            d, worst[1]["tpr_gap_ci_width"], worst[0]), flush=True)
    out["design_effect_sensitivity"] = {
        "assumption": "CI widths scale by sqrt(DEFF) under clustered/stratified survey sampling; "
                      "the cleaned release carries no design variables, so DEFF is assumed, not estimated",
        "by_deff": deff,
    }
    save_json(out, OUTPATH)

    # ============ H. SEVERITY-STRATIFIED SENSITIVITY ============
    print("\n=== H. Severity-stratified sensitivity (positive cases) ===", flush=True)
    sev = {}
    gh = Xte3["GenHlth"].values
    bmi = Xte3["BMI"].values
    pos = yv3 == 1
    strata = {
        "GenHlth 1-2 (good/very good)": pos & (gh <= 2),
        "GenHlth 3 (fair)": pos & (gh == 3),
        "GenHlth 4-5 (poor/very poor)": pos & (gh >= 4),
        "BMI < 30": pos & (bmi < 30),
        "BMI >= 30": pos & (bmi >= 30),
    }
    for name, msk in strata.items():
        if msk.sum() < 100: continue
        y_s = yv3[msk]; p_s = p_te[msk]
        try:
            auc = roc_auc_score(y_s, p_s)
        except Exception:
            auc = float("nan")
        sev[name] = {"n": int(msk.sum()), "within_stratum_auc": float(auc),
                     "mean_pred": float(p_s.mean())}
        print("  %-30s n=%6d within-stratum AUC=%.4f mean_pred=%.3f" % (
            name, sev[name]["n"], auc, sev[name]["mean_pred"]), flush=True)
    out["severity_sensitivity"] = sev

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
