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
OUTPATH = os.path.join(LOGS, "audit_results12.json")
ATTRS = ["Sex", "Age", "Income", "Education"]


def load_year(y):
    if y == 2015:
        return pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return pd.read_csv(os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % y))


def hgb(seed=42, balanced=True):
    p = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    if balanced:
        p["class_weight"] = "balanced"
    return HistGradientBoostingClassifier(**p)


def fit_safe(model, X, y, retries=4):
    import gc
    Xm = np.ascontiguousarray(X.to_numpy(dtype=np.float32)); ym = np.asarray(y); last = None
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


def gs(y, pred, groups, min_npos=20):
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        npos = int((y[m] == 1).sum()); nneg = int((y[m] == 0).sum())
        if npos < min_npos: continue
        per[g] = {"n": int(m.sum()), "n_pos": npos,
                  "tpr": float(((pred[m] == 1) & (y[m] == 1)).sum() / npos),
                  "fpr": float(((pred[m] == 1) & (y[m] == 0)).sum() / max(nneg, 1))}
    if not per: return {}, float("nan"), float("nan")
    t = [v["tpr"] for v in per.values()]; f = [v["fpr"] for v in per.values()]
    return per, float(max(t)-min(t)), float(max(f)-min(f))


def main():
    t0 = time.time(); out = {}
    df = load_year(2015); lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yv = yte.values
    m = hgb(42); fit_safe(m, Xtr, ytr)
    p_cal = predict_safe(m, Xcal); p_te = predict_safe(m, Xte)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_cal_iso = np.clip(iso.predict(p_cal), 0, 1); p_te_iso = np.clip(iso.predict(p_te), 0, 1)

    # global sensitivity-targeted threshold on calibrated calibration scores
    bt = None
    for t in np.linspace(0.02, 0.98, 193):
        pr = (p_cal_iso >= t).astype(int)
        s = ((pr == 1) & (ycal.values == 1)).sum() / max((ycal.values == 1).sum(), 1)
        if s >= 0.85: bt = t
    print("global sensitivity-targeted threshold (calibrated): %.3f" % bt, flush=True)

    def ev(y, prob, thr, X):
        pred = (prob >= thr).astype(int)
        e = {"threshold": float(thr),
             "overall_sensitivity": float(((pred == 1) & (y == 1)).sum()/max((y == 1).sum(),1)),
             "overall_specificity": float(((pred == 0) & (y == 0)).sum()/max((y == 0).sum(),1))}
        for g in ATTRS:
            _, tg, fg = gs(y, pred, X[g].values)
            e[g] = {"tpr_gap": tg, "fpr_gap": fg}
        key = X["Sex"].astype(str)+"|"+X["Age"].astype(str)
        _, tgi, fgi = gs(y, pred, key.values, min_npos=30)
        e["Sex_x_Age"] = {"tpr_gap": tgi, "fpr_gap": fgi}
        return e

    print("\n=== Global sensitivity-targeted policy: in-sample and out-of-time ===", flush=True)
    out["global_sens_targeted"] = {"2015_test": ev(yv, p_te_iso, bt, Xte)}
    e = out["global_sens_targeted"]["2015_test"]
    print("  2015 test: sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f | SexxAge %.4f" % (
        e["overall_sensitivity"], e["overall_specificity"], e["Sex"]["tpr_gap"], e["Age"]["tpr_gap"],
        e["Income"]["tpr_gap"], e["Education"]["tpr_gap"], e["Sex_x_Age"]["tpr_gap"]), flush=True)
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr); Xw = wy[FEATURES].reset_index(drop=True)
        yw = wy[TARGET].astype(int).values
        pw = np.clip(iso.predict(predict_safe(m, Xw)), 0, 1)
        # re-target the threshold on this wave to hold 0.85 sensitivity (transport test)
        bt_w = None
        for t in np.linspace(0.02, 0.98, 193):
            pr = (pw >= t).astype(int)
            s = ((pr == 1) & (yw == 1)).sum()/max((yw == 1).sum(),1)
            if s >= 0.85: bt_w = t
        e_w = ev(yw, pw, bt, Xw)          # transport the 2015 threshold
        e_wr = ev(yw, pw, bt_w, Xw)       # re-target on this wave
        out["global_sens_targeted"][str(yr)] = {"transported_threshold": e_w, "retargeted": e_wr}
        print("  %d transported thr=%.3f sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f" % (
            yr, bt, e_w["overall_sensitivity"], e_w["overall_specificity"], e_w["Sex"]["tpr_gap"],
            e_w["Age"]["tpr_gap"], e_w["Income"]["tpr_gap"], e_w["Education"]["tpr_gap"]), flush=True)
        print("      re-targeted thr=%.3f sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f" % (
            bt_w, e_wr["overall_sensitivity"], e_wr["overall_specificity"], e_wr["Sex"]["tpr_gap"],
            e_wr["Age"]["tpr_gap"], e_wr["Income"]["tpr_gap"], e_wr["Education"]["tpr_gap"]), flush=True)
    out["elapsed_s"] = round(time.time()-t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time()-t0), flush=True)


if __name__ == "__main__":
    main()
