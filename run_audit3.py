import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results3.json")

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
    return {"n": int(len(y)), "brier": float(brier_score_loss(y, p)),
            "ece": e, "mce": mc, "mean_pred": float(p.mean()),
            "observed_rate": float(y.mean())}

def main():
    t0 = time.time()
    out = {}
    df = pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    y_all = df[TARGET].astype(int).values

    # ---- split: train 64% / calib 16% / test 20% (stratified) ----
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    print("split: train=%d calib=%d test=%d" % (len(Xtr), len(Xcal), len(Xte)), flush=True)

    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                       class_weight="balanced", random_state=42)
    m.fit(Xtr, ytr)
    p_raw = m.predict_proba(Xte)[:, 1]
    yv = yte.values

    # ============== G. POST-HOC CALIBRATION ==============
    print("\n=== G. Post-hoc calibration (isotonic vs Platt, fit on held-out calib split) ===", flush=True)
    p_cal = m.predict_proba(Xcal)[:, 1]
    ycal_v = ycal.values
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal_v)
    p_iso = np.clip(iso.predict(p_raw), 0, 1)
    lp = LogisticRegression().fit(p_cal.reshape(-1, 1), ycal_v)
    p_plat = lp.predict_proba(p_raw.reshape(-1, 1))[:, 1]
    cal_res = {"uncalibrated": cal_block(yv, p_raw),
               "isotonic": cal_block(yv, p_iso),
               "platt": cal_block(yv, p_plat)}
    for k, v in cal_res.items():
        print("  %-13s brier=%.4f ECE=%.4f MCE=%.4f mean_pred=%.3f (obs=%.3f)" % (
            k, v["brier"], v["ece"], v["mce"], v["mean_pred"], v["observed_rate"]), flush=True)
    print("  AUC unchanged by monotone recalibration: %.4f" % roc_auc_score(yv, p_raw), flush=True)

    # per-group calibration before/after
    pg = {}
    for g in ["Sex", "Age", "Income", "Education"]:
        gm = Xte[g].values
        d = {}
        for name, pp in [("uncalibrated", p_raw), ("isotonic", p_iso)]:
            per = {}
            for gg in sorted(set(gm)):
                mask = gm == gg
                if mask.sum() < 50: continue
                per[str(gg)] = cal_block(yv[mask], pp[mask])
            d[name] = per
        pg[g] = d
    # summarize worst-group ECE before/after
    summ = {}
    for g, d in pg.items():
        for name, per in d.items():
            eces = [v["ece"] for v in per.values()]
            summ.setdefault(g, {})[name] = {"max_group_ece": float(max(eces)),
                                           "mean_group_ece": float(np.mean(eces))}
    for g, d in summ.items():
        print("  %-10s max-group ECE: raw=%.4f -> iso=%.4f" % (
            g, d["uncalibrated"]["max_group_ece"], d["isotonic"]["max_group_ece"]), flush=True)
    out["post_hoc_calibration"] = {"overall": cal_res, "by_group": pg,
                                   "group_summary": summ, "auc": float(roc_auc_score(yv, p_raw))}
    save_json(out, OUTPATH)
    print("  [checkpoint saved]", flush=True)

    # ============== H. MINIMUM-SUPPORT ROBUSTNESS FOR AGE GAP ==============
    print("\n=== H. Age gap robustness under minimum-support thresholds ===", flush=True)
    pred = (p_raw >= 0.5).astype(int)
    ages = Xte["Age"].values
    rows = []
    for a in sorted(set(ages)):
        mask = ages == a
        yt = yv[mask]; yp = pred[mask]
        npos = int(yt.sum()); nneg = int((yt == 0).sum())
        tpr = float(((yp == 1) & (yt == 1)).sum() / max(npos, 1))
        fpr = float(((yp == 1) & (yt == 0)).sum() / max(nneg, 1))
        rows.append({"age": int(a), "n": int(mask.sum()), "n_pos": npos, "n_neg": nneg,
                     "prevalence": float(yt.mean()), "tpr": tpr, "fpr": fpr,
                     "tpr_wilson_lo": None, "tpr_wilson_hi": None})
    # Wilson CI for TPR
    from math import sqrt
    def wilson(k, n, z=1.96):
        if n == 0: return (float("nan"), float("nan"))
        p = k / n; d = 1 + z*z/n
        c = (p + z*z/(2*n)) / d
        h = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
        return (max(0.0, c-h), min(1.0, c+h))
    for r in rows:
        lo, hi = wilson(round(r["tpr"]*r["n_pos"]), r["n_pos"])
        r["tpr_wilson_lo"], r["tpr_wilson_hi"] = lo, hi
    robust = {}
    for minpos in [1, 20, 50, 100, 500]:
        keep = [r for r in rows if r["n_pos"] >= minpos]
        if len(keep) < 2:
            robust[str(minpos)] = {"n_groups": len(keep), "tpr_gap": None}
            continue
        tprs = [r["tpr"] for r in keep]
        robust[str(minpos)] = {"n_groups": len(keep),
                               "tpr_gap": float(max(tprs) - min(tprs)),
                               "min_tpr": float(min(tprs)), "max_tpr": float(max(tprs)),
                               "widest_group_tpr_ci": float(max(r["tpr_wilson_hi"] - r["tpr_wilson_lo"] for r in keep))}
        print("  min n_pos >= %-4d : groups=%2d  TPR gap=%.4f  (widest within-group 95%% CI width=%.3f)" % (
            minpos, len(keep), robust[str(minpos)]["tpr_gap"],
            robust[str(minpos)]["widest_group_tpr_ci"]), flush=True)
    out["age_robustness"] = {"per_age": rows, "by_min_support": robust}
    save_json(out, OUTPATH)
    print("  [checkpoint saved]", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
