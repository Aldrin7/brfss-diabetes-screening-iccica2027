import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.feature_selection import mutual_info_classif

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results2.json")

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

def main():
    t0 = time.time()
    out = {}
    if os.path.exists(OUTPATH):
        try:
            out = json.load(open(OUTPATH))
            print("resuming with sections:", list(out.keys()), flush=True)
        except Exception:
            out = {}

    df = load_year(2015)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGET].astype(int),
                                           test_size=0.2, random_state=42, stratify=df[TARGET])
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    yte_v = yte.reset_index(drop=True).values

    # ================= D. HYPERPARAMETER SEARCH =================
    if "hyperparameter_search" not in out:
        print("=== D. Hyperparameter randomized search (3-fold CV, train only) ===", flush=True)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search_res = {}
        pipe_h = hgb(42)
        dist_h = {"learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15], "max_leaf_nodes": [15, 31, 63],
                  "max_iter": [100, 200, 300], "min_samples_leaf": [20, 40, 80],
                  "l2_regularization": [0.0, 0.1, 1.0]}
        rs_h = RandomizedSearchCV(pipe_h, dist_h, n_iter=20, cv=cv, scoring="roc_auc",
                                  random_state=42, n_jobs=1, refit=True)
        rs_h.fit(Xtr, ytr)
        search_res["HistGradientBoosting"] = {"best_params": {k: str(v) for k, v in rs_h.best_params_.items()},
                                              "best_cv_auc": float(rs_h.best_score_)}
        print("  HGB best cv AUC=%.4f %s" % (rs_h.best_score_, rs_h.best_params_), flush=True)
        pipe_r = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=1)
        dist_r = {"max_depth": [10, 16, 20], "min_samples_leaf": [10, 20, 40],
                  "max_features": ["sqrt", 0.5]}
        rs_r = RandomizedSearchCV(pipe_r, dist_r, n_iter=6, cv=cv, scoring="roc_auc",
                                  random_state=42, n_jobs=1, refit=True)
        rs_r.fit(Xtr, ytr)
        search_res["RandomForest"] = {"best_params": {k: str(v) for k, v in rs_r.best_params_.items()},
                                       "best_cv_auc": float(rs_r.best_score_)}
        print("  RF  best cv AUC=%.4f %s" % (rs_r.best_score_, rs_r.best_params_), flush=True)
        pipe_l = Pipeline([("s", StandardScaler()),
                           ("c", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
        dist_l = {"c__C": [0.01, 0.1, 1.0, 10.0, 100.0], "c__solver": ["lbfgs", "liblinear"],
                  "c__penalty": ["l2"]}
        rs_l = RandomizedSearchCV(pipe_l, dist_l, n_iter=10, cv=cv, scoring="roc_auc",
                                  random_state=42, n_jobs=1, refit=True)
        rs_l.fit(Xtr, ytr)
        search_res["LogisticRegression"] = {"best_params": {k: str(v) for k, v in rs_l.best_params_.items()},
                                            "best_cv_auc": float(rs_l.best_score_)}
        print("  LR  best cv AUC=%.4f %s" % (rs_l.best_score_, rs_l.best_params_), flush=True)
        out["hyperparameter_search"] = search_res
        save_json(out, OUTPATH)
        print("  [checkpoint saved]", flush=True)

    # ================= E. PER-WAVE THRESHOLD RE-OPTIMIZATION + DRIFT =================
    if "wave_drift" not in out:
        print("\n=== E. Per-wave threshold re-optimization + drift ===", flush=True)
        mfull = hgb(42); mfull.fit(df[FEATURES], df[TARGET].astype(int))
        p2015 = mfull.predict_proba(df[FEATURES])[:, 1]
        waves = {}
        for yr in [2017, 2019, 2021]:
            te = load_year(yr)
            Xw = te[FEATURES]; yw = te[TARGET].astype(int).values
            pw = mfull.predict_proba(Xw)[:, 1]
            best = 0.01
            for t in np.linspace(0.01, 0.99, 197):
                pr = (pw >= t).astype(int)
                s = ((pr == 1) & (yw == 1)).sum() / max((yw == 1).sum(), 1)
                if s >= 0.85: best = t
            pr = (pw >= best).astype(int)
            sens = ((pr == 1) & (yw == 1)).sum() / max((yw == 1).sum(), 1)
            spec = ((pr == 0) & (yw == 0)).sum() / max((yw == 0).sum(), 1)
            waves[str(yr)] = {"n": int(len(te)), "prevalence": float(yw.mean()),
                              "roc_auc": float(roc_auc_score(yw, pw)),
                              "thr_for_sens085": float(best), "sens": float(sens), "spec": float(spec),
                              "mean_pred": float(pw.mean()),
                              "pred_shift_vs_2015": float(pw.mean() - p2015.mean()),
                              "brier": float(brier_score_loss(yw, pw))}
            print("  %d: AUC=%.4f prev=%.3f thr(0.85)=%.3f sens=%.3f spec=%.3f brier=%.4f shift=%+.4f" % (
                yr, waves[str(yr)]["roc_auc"], waves[str(yr)]["prevalence"], best, sens, spec,
                waves[str(yr)]["brier"], waves[str(yr)]["pred_shift_vs_2015"]), flush=True)
        out["wave_drift"] = waves
        save_json(out, OUTPATH)
        print("  [checkpoint saved]", flush=True)

    # ================= F. ALTERNATIVE FEATURE SELECTORS =================
    if "feature_selectors" not in out:
        print("\n=== F. Feature selector comparison (k=10) ===", flush=True)
        import shap
        mt = hgb(42); mt.fit(Xtr, ytr)
        sv = shap.TreeExplainer(mt).shap_values(Xtr.sample(2000, random_state=42))
        if isinstance(sv, list): sv = sv[1]
        shap_rank = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
        mi = mutual_info_classif(Xtr.values, ytr.values, random_state=42, n_neighbors=5)
        mi_rank = [FEATURES[i] for i in np.argsort(mi)[::-1]]
        corr = np.array([abs(np.corrcoef(Xtr[c].values, ytr.values)[0, 1]) for c in FEATURES])
        corr_rank = [FEATURES[i] for i in np.argsort(corr)[::-1]]
        sel = {}
        for name, rank in [("shap", shap_rank), ("mutual_info", mi_rank), ("correlation", corr_rank)]:
            feats = rank[:10]
            mm = hgb(42); mm.fit(Xtr[feats], ytr)
            pr = mm.predict_proba(Xte[feats])[:, 1]
            sel["%s_k10" % name] = {"auc": float(roc_auc_score(yte_v, pr)), "features": feats}
            print("  %-14s k=10 AUC=%.4f" % (name, sel["%s_k10" % name]["auc"]), flush=True)
        rng = np.random.default_rng(42); aucs_r = []
        for _ in range(5):
            feats = list(rng.choice(FEATURES, 10, replace=False))
            mm = hgb(42); mm.fit(Xtr[feats], ytr)
            pr = mm.predict_proba(Xte[feats])[:, 1]
            aucs_r.append(float(roc_auc_score(yte_v, pr)))
        sel["random_k10"] = {"auc_mean": float(np.mean(aucs_r)), "auc_std": float(np.std(aucs_r, ddof=1))}
        print("  random k=10 AUC=%.4f +/- %.4f" % (np.mean(aucs_r), np.std(aucs_r, ddof=1)), flush=True)
        out["feature_selectors"] = sel
        save_json(out, OUTPATH)
        print("  [checkpoint saved]", flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
