import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results11.json")
ATTRS = ["Sex", "Age", "Income", "Education"]


def load_year(year):
    if year == 2015:
        return pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return pd.read_csv(os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % year))


def hgb(seed=42, balanced=True):
    p = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    if balanced:
        p["class_weight"] = "balanced"
    return HistGradientBoostingClassifier(**p)


def fit_safe(model, X, y, retries=4):
    import gc
    Xm = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    ym = np.asarray(y)
    last = None
    for a in range(retries):
        try:
            model.fit(Xm, ym); return model
        except Exception as e:
            last = e; gc.collect(); time.sleep(1.0 + a)
    raise last


def predict_safe(model, X, chunk=40000):
    Xm = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    out = np.empty(len(Xm), dtype=float)
    for i in range(0, len(Xm), chunk):
        out[i:i + chunk] = model.predict_proba(Xm[i:i + chunk])[:, 1]
    return out


def group_stats(y, pred, groups, min_npos=20):
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        npos = int((y[m] == 1).sum()); nneg = int((y[m] == 0).sum())
        if npos < min_npos: continue
        per[g] = {"n": int(m.sum()), "n_pos": npos,
                  "tpr": float(((pred[m] == 1) & (y[m] == 1)).sum() / npos),
                  "fpr": float(((pred[m] == 1) & (y[m] == 0)).sum() / max(nneg, 1))}
    if not per:
        return {}, float("nan"), float("nan")
    tprs = [v["tpr"] for v in per.values()]; fprs = [v["fpr"] for v in per.values()]
    return per, float(max(tprs) - min(tprs)), float(max(fprs) - min(fprs))


def fit_level_thresholds(prob_cal, Xcal, ycal, attr, target=0.85):
    """Threshold per level of `attr` achieving >= target sensitivity on the calibration split."""
    out = {}
    gv = Xcal[attr].values
    for lev in sorted(set(gv)):
        msk = gv == lev
        if (ycal[msk] == 1).sum() < 20: continue
        bt = None
        for t in np.linspace(0.02, 0.98, 193):
            pr = (prob_cal[msk] >= t).astype(int)
            s = ((pr == 1) & (ycal[msk] == 1)).sum() / max((ycal[msk] == 1).sum(), 1)
            if s >= target: bt = t
        if bt is not None: out[lev] = bt
    return out


def apply_policy(prob, X, thr_maps, policy):
    """policy: 'global' | attr name (use that attr's level threshold) | 'joint' (Sex x Age)"""
    if policy == "global":
        return (prob >= 0.5).astype(int)
    if policy == "joint":
        key = X["Sex"].astype(str) + "|" + X["Age"].astype(str)
        tm = thr_maps["joint"]
        pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            t = tm.get(key.iloc[i], 0.5)
            pred[i] = 1 if prob[i] >= t else 0
        return pred
    tm = thr_maps[policy]
    vals = X[policy].values
    pred = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        t = tm.get(vals[i], 0.5)
        pred[i] = 1 if prob[i] >= t else 0
    return pred


def evaluate(y, pred, X):
    e = {"overall_sensitivity": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
         "overall_specificity": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
         "overall_fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}
    for g in ATTRS:
        _, tg, fg = group_stats(y, pred, X[g].values)
        e[g] = {"tpr_gap": tg, "fpr_gap": fg}
    key = X["Sex"].astype(str) + "|" + X["Age"].astype(str)
    _, tgi, fgi = group_stats(y, pred, key.values, min_npos=30)
    e["Sex_x_Age"] = {"tpr_gap": tgi, "fpr_gap": fgi}
    return e


def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yv = yte.values

    m = hgb(42); fit_safe(m, Xtr, ytr)
    p_cal = predict_safe(m, Xcal)
    p_te = predict_safe(m, Xte)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_cal_iso = np.clip(iso.predict(p_cal), 0, 1)
    p_te_iso = np.clip(iso.predict(p_te), 0, 1)

    # threshold maps on calibrated calibration scores
    thr_maps = {a: fit_level_thresholds(p_cal_iso, Xcal, ycal.values, a) for a in ATTRS}
    # joint Sex x Age
    key_cal = Xcal["Sex"].astype(str) + "|" + Xcal["Age"].astype(str)
    joint = {}
    ycal_v = ycal.values
    for k in sorted(set(key_cal)):
        msk = key_cal == k
        if (ycal_v[msk] == 1).sum() < 20: continue
        bt = None
        for t in np.linspace(0.02, 0.98, 193):
            pr = (p_cal_iso[msk] >= t).astype(int)
            s = ((pr == 1) & (ycal_v[msk] == 1)).sum() / max((ycal_v[msk] == 1).sum(), 1)
            if s >= 0.85: bt = t
        if bt is not None: joint[k] = bt
    thr_maps["joint"] = joint

    policies = ["global", "Sex", "Age", "Income", "Education", "joint"]
    print("=== Threshold-policy comparison (calibrated scores, target 0.85 per level) ===", flush=True)
    res = {}
    for pol in policies:
        pred = apply_policy(p_te_iso, Xte, thr_maps, pol)
        e = evaluate(yv, pred, Xte)
        e["n_thresholds"] = len(thr_maps.get(pol, {})) if pol != "global" else 1
        res[pol] = e
        print("  %-10s sens=%.3f spec=%.3f | TPR gap: sex %.4f age %.4f inc %.4f edu %.4f | SexxAge %.4f" % (
            pol, e["overall_sensitivity"], e["overall_specificity"], e["Sex"]["tpr_gap"],
            e["Age"]["tpr_gap"], e["Income"]["tpr_gap"], e["Education"]["tpr_gap"],
            e["Sex_x_Age"]["tpr_gap"]), flush=True)
    out["policy_comparison_2015"] = res
    out["threshold_maps_sizes"] = {k: len(v) for k, v in thr_maps.items()}
    save_json(out, OUTPATH)

    # ---------- out-of-time ----------
    print("\n=== Out-of-time transfer of each policy (thresholds fitted on 2015) ===", flush=True)
    oot = {}
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr)
        Xw = wy[FEATURES].reset_index(drop=True)
        yw = wy[TARGET].astype(int).values
        pw = np.clip(iso.predict(predict_safe(m, Xw)), 0, 1)
        entry = {"auc": float(roc_auc_score(yw, pw))}
        for pol in policies:
            pred = apply_policy(pw, Xw, thr_maps, pol)
            entry[pol] = evaluate(yw, pred, Xw)
        oot[str(yr)] = entry
        print("  %d (AUC=%.4f):" % (yr, entry["auc"]), flush=True)
        for pol in policies:
            e = entry[pol]
            print("     %-10s sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f" % (
                pol, e["overall_sensitivity"], e["overall_specificity"], e["Sex"]["tpr_gap"],
                e["Age"]["tpr_gap"], e["Income"]["tpr_gap"], e["Education"]["tpr_gap"]), flush=True)
    out["policy_out_of_time"] = oot
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
