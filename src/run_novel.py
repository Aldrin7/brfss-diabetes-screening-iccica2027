import os, sys, json, gc, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import set_seed, ensure_dir, save_json
from preprocessing import FEATURES, TARGET
from fairness import compute_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(RESULTS, "logs")
ensure_dir(LOGS)

SHAP_ORDER = ["GenHlth","HighBP","Age","BMI","HighChol","Income","Sex","CholCheck",
              "MentHlth","HvyAlcoholConsump","HeartDiseaseorAttack","DiffWalk",
              "Education","Fruits","Stroke","PhysActivity","Veggies","Smoker",
              "AnyHealthcare","NoDocbcCost","PhysHlth"]

def load_year(year):
    p = os.path.join(MULTI, f"cdc_brfss_diabetes_{year}.csv")
    df = pd.read_csv(p)
    # normalize column names (already consistent)
    assert "Diabetes_binary" in df.columns
    return df

def train_eval(train_df, test_df, features=FEATURES, seed=42):
    set_seed(seed)
    Xtr = train_df[features]; ytr = train_df[TARGET].astype(int)
    Xte = test_df[features]; yte = test_df[TARGET].astype(int)
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                       class_weight="balanced", random_state=seed)
    m.fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:,1]
    # default 0.5
    pred = (prob >= 0.5).astype(int)
    mm = compute_metrics(yte.values, pred, prob)
    return m, mm, prob, yte

def main():
    t0 = time.time()
    out = {}
    # 1. Temporal external validation: train 2015, test 2017/2019/2021/2023
    print("=== Temporal external validation (train 2015) ===")
    tr = load_year(2015)
    temp = {}
    for yr in [2017, 2019, 2021]:
        te = load_year(yr)
        m, mm, prob, yte = train_eval(tr, te)
        temp[str(yr)] = {"n_test": len(te), "prevalence": float(yte.mean()),
                        "accuracy": mm["accuracy"], "f1": mm["f1"],
                        "roc_auc": mm["roc_auc"], "recall": mm["recall"],
                        "precision": mm["precision"]}
        print(f"  test {yr}: n={len(te)} prev={yte.mean():.3f} AUC={mm['roc_auc']:.4f} F1={mm['f1']:.4f} recall={mm['recall']:.4f}")
    out["temporal"] = temp

    # 2. Minimal questionnaire: top-k SHAP features, train/test on 2015
    print("\n=== Minimal questionnaire (top-k features, 2015) ===")
    df = load_year(2015)
    from sklearn.model_selection import train_test_split
    tr2, te2 = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    tr2 = tr2.reset_index(drop=True); te2 = te2.reset_index(drop=True)
    # SHAP ranking from training data only (avoid selection bias)
    import shap
    mtmp = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=42)
    mtmp.fit(tr2[FEATURES], tr2[TARGET].astype(int))
    Xs = tr2[FEATURES].sample(n=2000, random_state=42)
    sv = shap.TreeExplainer(mtmp).shap_values(Xs)
    if isinstance(sv, list): sv = sv[1]
    sv = np.array(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    shap_order = [FEATURES[i] for i in order]
    print("  train-only SHAP order:", shap_order[:8])
    mq = {}
    for k in [5, 8, 11, 15, 21]:
        feats = shap_order[:k]
        m, mm, prob, yte = train_eval(tr2, te2, features=feats)
        mq[str(k)] = {"features": feats, "n_features": k, "roc_auc": mm["roc_auc"],
                      "f1": mm["f1"], "accuracy": mm["accuracy"], "recall": mm["recall"]}
        print(f"  k={k}: AUC={mm['roc_auc']:.4f} F1={mm['f1']:.4f} acc={mm['accuracy']:.4f}")
    out["minimal_questionnaire"] = mq

    # 3. Fairness-aware threshold optimization on 2015 test
    print("\n=== Threshold optimization (2015 test, target sensitivity) ===")
    tr3, te3 = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
    tr3 = tr3.reset_index(drop=True); te3 = te3.reset_index(drop=True)
    m, mm, prob, yte = train_eval(tr3, te3)
    yte_arr = yte.values
    thr_res = {}
    for target_sens in [0.75, 0.80, 0.85, 0.90]:
        # find lowest threshold achieving >= target sensitivity
        best = None
        for t in np.linspace(0.01, 0.99, 197):
            pred = (prob >= t).astype(int)
            sens = ((pred==1)&(yte_arr==1)).sum() / max((yte_arr==1).sum(),1)
            if sens >= target_sens:
                best = (t, pred, sens)  # keep last (highest threshold) satisfying
        if best is None:
            thr_res[str(target_sens)] = {"achieved": False}; continue
        t, pred, sens = best
        spec = ((pred==0)&(yte_arr==0)).sum() / max((yte_arr==0).sum(),1)
        # sex TPR gap at this threshold
        sexes = te3["Sex"].values
        tprs = []
        for s in sorted(set(sexes)):
            mask = sexes==s
            tpr = ((pred[mask]==1)&(yte_arr[mask]==1)).sum() / max((yte_arr[mask]==1).sum(),1)
            tprs.append(tpr)
        thr_res[str(target_sens)] = {
            "threshold": float(t), "sensitivity": float(sens), "specificity": float(spec),
            "f1": float(f1_score(yte_arr, pred, zero_division=0)),
            "accuracy": float(accuracy_score(yte_arr, pred)),
            "sex_tpr_gap": float(max(tprs)-min(tprs)),
            "positive_rate": float(pred.mean()),
        }
        print(f"  target sens {target_sens}: thr={t:.3f} sens={sens:.3f} spec={spec:.3f} f1={thr_res[str(target_sens)]['f1']:.3f} sex_tpr_gap={thr_res[str(target_sens)]['sex_tpr_gap']:.3f}")
    out["threshold_optimization"] = thr_res

    out["elapsed_s"] = round(time.time()-t0, 1)
    save_json(out, os.path.join(LOGS, "novel_results.json"))
    print("\nTotal elapsed: %.1f s" % (time.time()-t0))
    print("saved novel_results.json")

if __name__ == "__main__":
    main()
