import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results4.json")

def main():
    t0 = time.time()
    out = {}
    df = pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    X = df[FEATURES]; y = df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yv = yte.values

    models = {
        "LogisticRegression": Pipeline([("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
            max_leaf_nodes=31, class_weight="balanced", random_state=42),
    }
    fitted = {}
    probs = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        probs[name] = m.predict_proba(Xte)[:, 1]
        print("  fitted %s" % name, flush=True)

    # ============ I. PAIRED BOOTSTRAP AUC COMPARISON (DeLong-equivalent) ============
    print("\n=== I. Paired bootstrap AUC comparison (5 seeds x 2000 resamples) ===", flush=True)
    rng = np.random.default_rng(42)
    n = len(yv)
    boot = {k: [] for k in models}
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        for k, p in probs.items():
            boot[k].append(roc_auc_score(yv[idx], p[idx]))
    boot = {k: np.array(v) for k, v in boot.items()}
    test = {}
    for k, p in probs.items():
        test[k] = float(roc_auc_score(yv, p))
    pairs = {}
    ks = list(models.keys())
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            a, b = ks[i], ks[j]
            d = boot[a] - boot[b]
            obs = test[a] - test[b]
            ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
            pairs["%s vs %s" % (a, b)] = {"auc_a": test[a], "auc_b": test[b],
                                          "delta_auc": obs, "ci95": ci, "p_value": float(min(1.0, pval)),
                                          "significant": bool(ci[0] > 0 or ci[1] < 0)}
            print("  %-42s dAUC=%+.4f CI[%+.4f,%+.4f] p=%.4f %s" % (
                "%s vs %s" % (a, b), obs, ci[0], ci[1], min(1.0, pval),
                "SIG" if (ci[0] > 0 or ci[1] < 0) else "ns"), flush=True)
    out["auc_pairwise"] = {"test_auc": test, "pairs": pairs, "n_boot": 2000}

    # ============ J. SCENARIO ANALYSIS (bounded perturbation) ============
    print("\n=== J. Scenario analysis (bounded perturbation, HGB) ===", flush=True)
    m = fitted["HistGradientBoosting"]
    base = probs["HistGradientBoosting"]
    base_mean = float(base.mean())
    scen = {}
    # describe perturbations bounded by historical feature range
    scen_defs = {
        "BMI +5 (1 SD approx)": ("BMI", lambda v: v + 5),
        "BMI +10": ("BMI", lambda v: v + 10),
        "Smoker 0->1": ("Smoker", lambda v: 1),
        "PhysActivity 1->0": ("PhysActivity", lambda v: 0),
        "HvyAlcoholConsump 0->1": ("HvyAlcoholConsump", lambda v: 1),
        "GenHlth +1 (worse)": ("GenHlth", lambda v: v + 1),
    }
    for label, (col, fn) in scen_defs.items():
        Xp = Xte.copy()
        lo, hi = float(Xp[col].min()), float(Xp[col].max())
        Xp[col] = fn(Xp[col].astype(float))
        # clip to historical range
        Xp[col] = Xp[col].clip(lo, hi)
        n_clipped = float((fn(Xp[col].astype(float)) != Xp[col]).sum()) if False else 0
        pp = m.predict_proba(Xp)[:, 1]
        d = pp - base
        # paired bootstrap CI of mean shift
        rng2 = np.random.default_rng(7)
        idx = rng2.integers(0, n, 500)
        shifts = np.array([d[i].mean() for i in idx])
        ci = [float(np.percentile(shifts, 2.5)), float(np.percentile(shifts, 97.5))]
        scen[label] = {"column": col, "base_mean_prob": base_mean,
                       "new_mean_prob": float(pp.mean()),
                       "mean_abs_shift": float(d.mean()),
                       "mean_prob_change": float(pp.mean() - base_mean),
                       "ci95": ci,
                       "share_increased": float((d > 0).mean()),
                       "feature_range": [lo, hi]}
        print("  %-24s mean prob %+.4f (CI %+.4f,%+.4f)  share increased=%.3f" % (
            label, scen[label]["mean_prob_change"], ci[0], ci[1], scen[label]["share_increased"]), flush=True)
    out["scenario_analysis"] = scen
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
