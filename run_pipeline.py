import os, sys, json, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import set_seed, ensure_dir, save_json
from preprocessing import load_raw, split_data, FEATURES, TARGET, SENSITIVE, CONTINUOUS
from models import build_models
from fairness import compute_metrics, group_metrics, fairness_gaps

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
for d in (RESULTS, LOGS, TABLES, FIGS):
    ensure_dir(d)

SEEDS = [42, 7, 123, 2024, 99]
TEST_SIZE = 0.2

def run_one_seed(seed):
    set_seed(seed)
    df = load_raw(DATA)
    X_tr, X_te, y_tr, y_te = split_data(df, seed=seed, test_size=TEST_SIZE)
    models = build_models(seed=seed)
    out = {"seed": seed, "n_train": len(X_tr), "n_test": len(X_te),
           "test_prevalence": float(y_te.mean()), "models": {}}
    for name, model in models.items():
        t0 = time.time()
        gc.collect()
        model.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        m = compute_metrics(y_te.values, pred, prob)
        m["train_time_s"] = round(dt, 2)
        # fairness by sex, age, income, education (use test groups)
        for grp in ["Sex", "Age", "Income", "Education"]:
            gm = group_metrics(y_te.values, pred, X_te[grp].values)
            m["fairness_" + grp] = gm
            m["fairness_" + grp + "_gaps"] = fairness_gaps(gm)
        out["models"][name] = m
        print(f"  seed {seed} {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m.get('roc_auc',float('nan')):.4f} t={dt:.1f}s")
    return out

def run_ablation(seed=42):
    set_seed(seed)
    df = load_raw(DATA)
    X_tr, X_te, y_tr, y_te = split_data(df, seed=seed, test_size=TEST_SIZE)
    groups = {
        "demographic_only": ["Sex", "Age", "Education", "Income"],
        "plus_clinical": ["Sex", "Age", "Education", "Income", "HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack", "GenHlth", "BMI"],
        "plus_lifestyle": ["Sex", "Age", "Education", "Income", "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "PhysHlth", "DiffWalk", "AnyHealthcare", "NoDocbcCost"],
        "full": FEATURES,
    }
    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {}
    for gname, feats in groups.items():
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced", random_state=seed)
        m.fit(X_tr[feats], y_tr)
        pred = m.predict(X_te[feats])
        prob = m.predict_proba(X_te[feats])[:, 1]
        mm = compute_metrics(y_te.values, pred, prob)
        res[gname] = {"n_features": len(feats), **mm}
        print(f"  ablation {gname}: auc={mm.get('roc_auc',float('nan')):.4f} f1={mm['f1']:.4f}")
    return res

def main():
    t0 = time.time()
    all_runs = []
    for s in SEEDS:
        print(f"\n=== Seed {s} ===")
        all_runs.append(run_one_seed(s))
    print("\n=== Ablation (seed 42) ===")
    abl = run_ablation(42)
    # aggregate
    agg = {}
    for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]:
        accs = [r["models"][name]["accuracy"] for r in all_runs]
        f1s = [r["models"][name]["f1"] for r in all_runs]
        aucs = [r["models"][name].get("roc_auc", float("nan")) for r in all_runs]
        agg[name] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs, ddof=1)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
            "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs, ddof=1)),
            "n_seeds": len(SEEDS),
        }
    save_json({"seeds": SEEDS, "runs": all_runs, "aggregate": agg, "ablation": abl,
               "elapsed_s": round(time.time()-t0, 1)}, os.path.join(LOGS, "results.json"))
    # print summary
    print("\n=== AGGREGATE (mean +/- std over %d seeds) ===" % len(SEEDS))
    for name, a in agg.items():
        print(f"{name}: acc={a['accuracy_mean']:.4f}+/-{a['accuracy_std']:.4f} f1={a['f1_mean']:.4f}+/-{a['f1_std']:.4f} auc={a['auc_mean']:.4f}+/-{a['auc_std']:.4f}")
    print("\nAblation:")
    for g, a in abl.items():
        print(f"  {g} (n={a['n_features']}): auc={a['roc_auc']:.4f} f1={a['f1']:.4f} acc={a['accuracy']:.4f}")
    print("\nTotal elapsed: %.1f s" % (time.time()-t0))

if __name__ == "__main__":
    main()
