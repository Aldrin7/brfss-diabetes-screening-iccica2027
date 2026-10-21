import os, sys, json, time, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import set_seed, ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
ensure_dir(LOGS)

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

# ---------- calibration ----------
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

def calibration_block(y, p, groups=None):
    out = {"n": int(len(y)),
           "brier": float(brier_score_loss(y, p)),
           "ece": ece_mce(y, p)[0], "mce": ece_mce(y, p)[1],
           "mean_pred": float(p.mean()), "observed_rate": float(y.mean())}
    if groups is not None:
        per = {}
        for g in sorted(set(groups)):
            m = groups == g
            if m.sum() < 50: continue
            e, mc = ece_mce(y[m], p[m])
            per[str(g)] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                           "prevalence": float(y[m].mean()),
                           "brier": float(brier_score_loss(y[m], p[m])),
                           "ece": e, "mce": mc}
        out["by_group"] = per
    return out

# ---------- fairness ----------
def fairness_block(y, p, thr, groups, min_n=50):
    pred = (p >= thr).astype(int)
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < min_n: continue
        yt = y[m]; yp = pred[m]
        npos = int(yt.sum()); nneg = int((yt==0).sum())
        per[str(g)] = {
            "n": int(m.sum()), "n_pos": npos, "n_neg": nneg,
            "prevalence": float(yt.mean()),
            "tpr": float(((yp==1)&(yt==1)).sum()/max(npos,1)),
            "fpr": float(((yp==1)&(yt==0)).sum()/max(nneg,1)),
            "sel_rate": float(yp.mean()),
            "brier": float(brier_score_loss(yt, p[m])),
        }
    def gap(key):
        vals = [v[key] for v in per.values()]
        return float(max(vals)-min(vals)) if vals else float("nan")
    return {"per_group": per,
            "tpr_gap": gap("tpr"), "fpr_gap": gap("fpr"),
            "eod": gap("tpr"), "sel_rate_gap": gap("sel_rate"),
            "min_group_n": int(min(v["n"] for v in per.values())) if per else 0,
            "min_group_n_pos": int(min(v["n_pos"] for v in per.values())) if per else 0}

def bootstrap_ci(y, p, groups, thr, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    n = len(y)
    tpr_gaps = []; fpr_gaps = []; aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fb = fairness_block(y[idx], p[idx], thr, groups[idx])
        if not np.isnan(fb["tpr_gap"]):
            tpr_gaps.append(fb["tpr_gap"]); fpr_gaps.append(fb["fpr_gap"])
        try:
            aucs.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            pass
    def ci(v):
        if not v: return [float("nan")]*2
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"tpr_gap_ci95": ci(tpr_gaps), "fpr_gap_ci95": ci(fpr_gaps),
            "auc_ci95": ci(aucs), "n_boot": n_boot}

def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    y_all = df[TARGET].astype(int).values

    # ================= A. LEAKAGE / DUPLICATE SENSITIVITY =================
    print("=== A. Duplicate-profile leakage sensitivity ===", flush=True)
    dup_mask = df.duplicated(subset=FEATURES, keep="first")
    out["duplicates"] = {"n_rows": int(len(df)),
                          "n_duplicate_rows": int(dup_mask.sum()),
                          "pct_duplicate": float(dup_mask.mean()),
                          "n_unique_profiles": int(len(df.drop_duplicates(subset=FEATURES)))}
    print("  duplicate rows:", int(dup_mask.sum()), "=%.2f%%" % (100*dup_mask.mean()))

    # A1: train on deduplicated, test on deduplicated
    dedup = df[~dup_mask].reset_index(drop=True)
    Xd = dedup[FEATURES]; yd = dedup[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    ded = {}
    for name, mk in [("LogisticRegression", lambda: Pipeline([("s", StandardScaler()),
                        ("c", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])),
                     ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, max_depth=20,
                        min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=1)),
                     ("HistGradientBoosting", lambda: hgb(42))]:
        m = mk(); m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:,1]
        ded[name] = {"roc_auc": float(roc_auc_score(yte, pr)),
                     "brier": float(brier_score_loss(yte, pr))}
        print("  dedup %-22s AUC=%.4f brier=%.4f" % (name, ded[name]["roc_auc"], ded[name]["brier"]))
    out["dedup_refit"] = {"n": int(len(dedup)), "models": ded}

    # A2: grouped split on full data (identical covariate vectors never cross folds)
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False).values
    X = df[FEATURES].values
    aucs = {"dedup_test": [], "grouped_cv": []}
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for tr_i, te_i in gkf.split(X, y_all, groups):
        m = hgb(42); m.fit(X[tr_i], y_all[tr_i])
        oof[te_i] = m.predict_proba(X[te_i])[:,1]
    aucs["grouped_cv"] = float(roc_auc_score(y_all, oof))
    print("  grouped 5-fold CV (duplicates isolated) AUC=%.4f" % aucs["grouped_cv"])
    # comparable plain stratified CV for reference
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof2 = np.zeros(len(df))
    for tr_i, te_i in skf.split(X, y_all):
        m = hgb(42); m.fit(X[tr_i], y_all[tr_i])
        oof2[te_i] = m.predict_proba(X[te_i])[:,1]
    aucs["stratified_cv"] = float(roc_auc_score(y_all, oof2))
    print("  plain stratified 5-fold CV AUC=%.4f" % aucs["stratified_cv"])
    out["leakage_sensitivity"] = {
        "grouped_cv_auc": aucs["grouped_cv"], "stratified_cv_auc": aucs["stratified_cv"],
        "delta": aucs["stratified_cv"] - aucs["grouped_cv"],
        "dedup_test_auc_hgb": ded["HistGradientBoosting"]["roc_auc"],
        "original_test_auc_hgb": 0.8269}
    print("  delta (stratified - grouped) = %.4f" % out["leakage_sensitivity"]["delta"])

    # ================= B. CALIBRATION =================
    print("\n=== B. Calibration ===", flush=True)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGET].astype(int),
                                           test_size=0.2, random_state=42, stratify=df[TARGET])
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    yte_v = yte.values
    m = hgb(42); m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:,1]
    cal = {"overall": calibration_block(yte_v, p)}
    for g in ["Sex", "Age", "Income", "Education"]:
        cal["group_" + g] = calibration_block(yte_v, p, Xte[g].values)
    out["calibration"] = cal
    print("  overall: brier=%.4f ECE=%.4f MCE=%.4f (mean_pred=%.3f obs=%.3f)" % (
        cal["overall"]["brier"], cal["overall"]["ece"], cal["overall"]["mce"],
        cal["overall"]["mean_pred"], cal["overall"]["observed_rate"]))
    print("  sex ECE=%.4f | age ECE=%.4f | income ECE=%.4f | edu ECE=%.4f" % (
        cal["group_Sex"]["ece"], cal["group_Age"]["ece"],
        cal["group_Income"]["ece"], cal["group_Education"]["ece"]))
    np.save(os.path.join(LOGS, "_calib_probs.npy"), p)
    Xte[["Sex","Age","Income","Education"]].to_csv(os.path.join(LOGS, "_calib_groups.csv"), index=False)

    # ================= C. FAIRNESS DIAGNOSTICS =================
    print("\n=== C. Fairness diagnostics (supports, EOD, bootstrap CIs) ===", flush=True)
    fair = {}
    for g in ["Sex", "Age", "Income", "Education"]:
        fb = fairness_block(yte_v, p, 0.5, Xte[g].values)
        bs = bootstrap_ci(yte_v, p, Xte[g].values, 0.5, n_boot=300, seed=42)
        fair[g] = {"fairness": fb, "bootstrap": bs}
        print("  %-10s TPRgap=%.4f CI[%.4f,%.4f]  FPRgap=%.4f CI[%.4f,%.4f]  min_n=%d min_npos=%d" % (
            g, fb["tpr_gap"], bs["tpr_gap_ci95"][0], bs["tpr_gap_ci95"][1],
            fb["fpr_gap"], bs["fpr_gap_ci95"][0], bs["fpr_gap_ci95"][1],
            fb["min_group_n"], fb["min_group_n_pos"]))
    out["fairness_diagnostics"] = fair

    # ================= D. HYPERPARAMETER SEARCH =================
    print("\n=== D. Hyperparameter randomized search ===", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search_res = {}
    # HGB
    pipe_h = hgb(42)
    dist_h = {"learning_rate": [0.03,0.05,0.08,0.1,0.15], "max_leaf_nodes": [15,31,63],
              "max_iter": [100,200,300], "min_samples_leaf": [20,40,80],
              "l2_regularization": [0.0,0.1,1.0]}
    rs_h = RandomizedSearchCV(pipe_h, dist_h, n_iter=20, cv=cv, scoring="roc_auc",
                              random_state=42, n_jobs=1, refit=True)
    rs_h.fit(Xtr, ytr)
    search_res["HistGradientBoosting"] = {"best_params": {k: str(v) for k,v in rs_h.best_params_.items()},
                                          "best_cv_auc": float(rs_h.best_score_)}
    print("  HGB best cv AUC=%.4f params=%s" % (rs_h.best_score_, rs_h.best_params_))
    # RF
    pipe_r = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1)
    dist_r = {"max_depth": [8,12,16,20,None], "min_samples_leaf": [5,10,20,40],
              "max_features": ["sqrt", 0.5, 0.8]}
    rs_r = RandomizedSearchCV(pipe_r, dist_r, n_iter=10, cv=cv, scoring="roc_auc",
                              random_state=42, n_jobs=1, refit=True)
    rs_r.fit(Xtr, ytr)
    search_res["RandomForest"] = {"best_params": {k: str(v) for k,v in rs_r.best_params_.items()},
                                   "best_cv_auc": float(rs_r.best_score_)}
    print("  RF  best cv AUC=%.4f params=%s" % (rs_r.best_score_, rs_r.best_params_))
    # LR
    pipe_l = Pipeline([("s", StandardScaler()),
                       ("c", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
    dist_l = {"c__C": [0.01,0.1,1.0,10.0,100.0], "c__solver": ["lbfgs","liblinear"],
              "c__penalty": ["l2"]}
    rs_l = RandomizedSearchCV(pipe_l, dist_l, n_iter=10, cv=cv, scoring="roc_auc",
                              random_state=42, n_jobs=1, refit=True)
    rs_l.fit(Xtr, ytr)
    search_res["LogisticRegression"] = {"best_params": {k: str(v) for k,v in rs_l.best_params_.items()},
                                        "best_cv_auc": float(rs_l.best_score_)}
    print("  LR  best cv AUC=%.4f params=%s" % (rs_l.best_score_, rs_l.best_params_))
    out["hyperparameter_search"] = search_res

    # ================= E. PER-WAVE THRESHOLD RE-OPTIMIZATION + DRIFT =================
    print("\n=== E. Per-wave threshold re-optimization + drift ===", flush=True)
    mfull = hgb(42); mfull.fit(df[FEATURES], df[TARGET].astype(int))
    waves = {}
    for yr in [2017, 2019, 2021]:
        te = load_year(yr)
        Xw = te[FEATURES]; yw = te[TARGET].astype(int).values
        pw = mfull.predict_proba(Xw)[:,1]
        best = None
        for t in np.linspace(0.01, 0.99, 197):
            pr = (pw >= t).astype(int)
            s = ((pr==1)&(yw==1)).sum()/max((yw==1).sum(),1)
            if s >= 0.85: best = t
        if best is None: best = 0.01
        pr = (pw >= best).astype(int)
        sens = ((pr==1)&(yw==1)).sum()/max((yw==1).sum(),1)
        spec = ((pr==0)&(yw==0)).sum()/max((yw==0).sum(),1)
        # drift vs 2015
        d_mean = float(pw.mean() - p.mean())
        waves[str(yr)] = {"n": int(len(te)), "prevalence": float(yw.mean()),
                          "roc_auc": float(roc_auc_score(yw, pw)),
                          "thr_for_sens085": float(best), "sens": float(sens), "spec": float(spec),
                          "mean_pred": float(pw.mean()),
                          "pred_shift_vs_2015": d_mean,
                          "brier": float(brier_score_loss(yw, pw))}
        print("  %d: AUC=%.4f prev=%.3f thr(0.85)=%.3f sens=%.3f spec=%.3f brier=%.4f predshift=%+.4f" % (
            yr, waves[str(yr)]["roc_auc"], waves[str(yr)]["prevalence"], best, sens, spec,
            waves[str(yr)]["brier"], d_mean))
    out["wave_drift"] = waves

    # ================= F. ALTERNATIVE FEATURE SELECTORS =================
    print("\n=== F. Feature selector comparison (k=10) ===", flush=True)
    import shap
    mt = hgb(42); mt.fit(Xtr, ytr)
    sv = shap.TreeExplainer(mt).shap_values(Xtr.sample(2000, random_state=42))
    if isinstance(sv, list): sv = sv[1]
    shap_rank = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
    mi = mutual_info_classif(Xtr.values, ytr.values, random_state=42, n_neighbors=5)
    mi_rank = [FEATURES[i] for i in np.argsort(mi)[::-1]]
    corr = np.array([abs(np.corrcoef(Xtr[c].values, ytr.values)[0,1]) for c in FEATURES])
    corr_rank = [FEATURES[i] for i in np.argsort(corr)[::-1]]
    sel = {}
    for k in [10]:
        for name, rank in [("shap", shap_rank), ("mutual_info", mi_rank), ("correlation", corr_rank)]:
            feats = rank[:k]
            mm = hgb(42); mm.fit(Xtr[feats], ytr)
            pr = mm.predict_proba(Xte[feats])[:,1]
            sel["%s_k%d" % (name, k)] = {"auc": float(roc_auc_score(yte.values, pr)),
                                         "features": feats}
            print("  %-14s k=%d AUC=%.4f" % (name, k, sel["%s_k%d" % (name, k)]["auc"]))
    # random baseline
    rng = np.random.default_rng(42); aucs_r = []
    for _ in range(5):
        feats = list(rng.choice(FEATURES, 10, replace=False))
        mm = hgb(42); mm.fit(Xtr[feats], ytr)
        pr = mm.predict_proba(Xte[feats])[:,1]
        aucs_r.append(float(roc_auc_score(yte.values, pr)))
    sel["random_k10"] = {"auc_mean": float(np.mean(aucs_r)), "auc_std": float(np.std(aucs_r, ddof=1))}
    print("  random k=10 AUC=%.4f +/- %.4f" % (np.mean(aucs_r), np.std(aucs_r, ddof=1)))
    out["feature_selectors"] = sel

    out["elapsed_s"] = round(time.time()-t0, 1)
    save_json(out, os.path.join(LOGS, "audit_results.json"))
    print("\nTotal elapsed: %.1f s" % (time.time()-t0))
    print("saved audit_results.json")

if __name__ == "__main__":
    main()
