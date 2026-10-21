import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results13.json")
ATTRS = ["Sex", "Age", "Income", "Education"]


def load_year(y):
    if y == 2015:
        return pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return pd.read_csv(os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % y))


def hgb(seed=42):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=seed)


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


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        e += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    t0 = time.time(); out = {}
    df = load_year(2015); lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yv = yte.reset_index(drop=True).values
    m = hgb(42); fit_safe(m, Xtr, ytr)
    p_cal = predict_safe(m, Xcal); p_te = predict_safe(m, Xte)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_cal_iso = np.clip(iso.predict(p_cal), 0, 1); p_te_iso = np.clip(iso.predict(p_te), 0, 1)

    # ============ P. DIAGNOSE THE THRESHOLD-POLICY FAILURE ============
    print("=== P. Why attribute-specific thresholds fail out of time ===", flush=True)
    # per-level 2015 thresholds (target 0.85 on calibrated calibration scores)
    thr = {}
    for a in ATTRS:
        for lev in sorted(set(Xcal[a].values)):
            msk = Xcal[a].values == lev
            if (ycal.values[msk] == 1).sum() < 20: continue
            bt = None
            for t in np.linspace(0.02, 0.98, 193):
                pr = (p_cal_iso[msk] >= t).astype(int)
                s = ((pr == 1) & (ycal.values[msk] == 1)).sum() / max((ycal.values[msk] == 1).sum(), 1)
                if s >= 0.85: bt = t
            if bt is not None: thr[(a, lev)] = bt
    diag = {}
    for a in ATTRS:
        per = {}
        for lev in sorted(set(Xcal[a].values)):
            msk_c = Xcal[a].values == lev
            if (ycal.values[msk_c] == 1).sum() < 20: continue
            if (a, lev) not in thr: continue
            row = {"n_cal": int(msk_c.sum()), "n_cal_pos": int((ycal.values[msk_c] == 1).sum()),
                   "prev_2015": float(ycal.values[msk_c].mean()),
                   "mean_score_2015": float(p_cal_iso[msk_c].mean()),
                   "thr_2015": float(thr[(a, lev)])}
            for yr in [2017, 2019, 2021]:
                wy = load_year(yr)
                msk_w = wy[a].values == lev
                yw = wy[TARGET].astype(int).values[msk_w]
                pw = np.clip(iso.predict(predict_safe(m, wy[FEATURES].reset_index(drop=True).iloc[msk_w])), 0, 1)
                bt = None
                for t in np.linspace(0.02, 0.98, 193):
                    pr = (pw >= t).astype(int)
                    s = ((pr == 1) & (yw == 1)).sum() / max((yw == 1).sum(), 1)
                    if s >= 0.85: bt = t
                pred_2015thr = (pw >= row["thr_2015"]).astype(int)
                row["w%d" % yr] = {
                    "n": int(msk_w.sum()), "n_pos": int((yw == 1).sum()),
                    "prev": float(yw.mean()),
                    "prev_shift": float(yw.mean() - row["prev_2015"]),
                    "mean_score": float(pw.mean()),
                    "score_shift": float(pw.mean() - row["mean_score_2015"]),
                    "sens_at_2015_thr": float(((pred_2015thr == 1) & (yw == 1)).sum() / max((yw == 1).sum(), 1)),
                    "thr_retargeted": (float(bt) if bt is not None else None),
                    "thr_drift": (float(bt - row["thr_2015"]) if bt is not None else None),
                    "ece_calibrated": ece(yw, pw),
                }
            per[str(lev)] = row
        diag[a] = per
        # summarise
        d2021 = [v.get("w2021", {}) for v in per.values()]
        sens = [v["sens_at_2015_thr"] for v in d2021 if "sens_at_2015_thr" in v]
        drift = [v["thr_drift"] for v in d2021 if v.get("thr_drift") is not None]
        print("  %-10s levels=%2d | 2021 sens at 2015 thr: min=%.3f max=%.3f | threshold drift: min=%+.3f max=%+.3f" % (
            a, len(per), min(sens), max(sens), min(drift), max(drift)), flush=True)
    out["threshold_failure_diagnosis"] = diag
    save_json(out, OUTPATH)

    # ============ Q. TUNED XGBOOST (performance ceiling check) ============
    print("\n=== Q. Tuned XGBoost for a performance-ceiling check ===", flush=True)
    try:
        import xgboost as xgb
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=42, n_jobs=2, tree_method="hist")
        dist = {"max_depth": [3, 5, 7, 9], "learning_rate": [0.03, 0.05, 0.1],
                "min_child_weight": [1, 5, 20], "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}
        rs = RandomizedSearchCV(clf, dist, n_iter=12, cv=cv, scoring="roc_auc",
                                 random_state=42, n_jobs=1, refit=True)
        rs.fit(Xtr, ytr)
        p_xgb = rs.predict_proba(Xte)[:, 1]
        xgb_res = {"best_params": {k: str(v) for k, v in rs.best_params_.items()},
                   "best_cv_auc": float(rs.best_score_),
                   "test_roc_auc": float(roc_auc_score(yv, p_xgb)),
                   "test_auprc": float(average_precision_score(yv, p_xgb)),
                   "hgb_test_roc_auc": float(roc_auc_score(yv, p_te)),
                   "hgb_test_auprc": float(average_precision_score(yv, p_te))}
        print("  XGB tuned: best CV AUC=%.4f | test AUC=%.4f AUPRC=%.4f (HGB: %.4f / %.4f)" % (
            xgb_res["best_cv_auc"], xgb_res["test_roc_auc"], xgb_res["test_auprc"],
            xgb_res["hgb_test_roc_auc"], xgb_res["hgb_test_auprc"]), flush=True)
        out["xgboost_tuned"] = xgb_res
    except Exception as e:
        print("  xgboost failed:", e, flush=True)
        out["xgboost_tuned"] = {"error": str(e)}
    save_json(out, OUTPATH)

    # ============ R. PERMUTATION IMPORTANCE vs SHAP ============
    print("\n=== R. Permutation importance vs SHAP ranking ===", flush=True)
    pi = permutation_importance(m, Xte, yv, n_repeats=5, random_state=42,
                                scoring="roc_auc", n_jobs=1)
    perm_rank = [FEATURES[i] for i in np.argsort(pi.importances_mean)[::-1]]
    import shap
    sv = shap.TreeExplainer(m).shap_values(Xte.sample(2000, random_state=42))
    if isinstance(sv, list): sv = sv[1]
    shap_rank = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
    rho = float(spearmanr([perm_rank.index(f) for f in shap_rank],
                           list(range(len(shap_rank)))).statistic)
    top10 = len(set(perm_rank[:10]) & set(shap_rank[:10]))
    print("  permutation top-10: %s" % ", ".join(perm_rank[:10]), flush=True)
    print("  SHAP        top-10: %s" % ", ".join(shap_rank[:10]), flush=True)
    print("  Spearman(rank) = %.3f | top-10 overlap = %d/10" % (rho, top10), flush=True)
    out["shap_vs_permutation"] = {"permutation_ranking": perm_rank, "shap_ranking": shap_rank,
                                  "spearman": rho, "top10_overlap": top10}
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
