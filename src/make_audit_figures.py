import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
FIGS = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

D = {}
for f in sorted(glob.glob(os.path.join(LOGS, "audit_results*.json"))):
    try:
        D.update(json.load(open(f)))
    except Exception as e:
        print("skip", f, e)
print("loaded sections:", list(D.keys()))

df = pd.read_csv(os.path.join(ROOT, "data", "raw", "diabetes_binary_health_indicators_BRFSS2015.csv"))
from sklearn.model_selection import train_test_split as tts
Xrest, Xte, yrest, yte = tts(df, df["Diabetes_binary"].astype(int), test_size=0.20, random_state=42, stratify=df["Diabetes_binary"])
Xtr, Xcal, ytr, ycal = tts(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
Xte = Xte.reset_index(drop=True); yv = yte.astype(int).values

m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                   class_weight="balanced", random_state=42)
m.fit(Xtr, ytr)
p_raw = m.predict_proba(Xte)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m.predict_proba(Xcal)[:, 1], ycal.values)
p_iso = np.clip(iso.predict(p_raw), 0, 1)

def reliability(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        msk = idx == b
        if msk.sum() < 30: continue
        xs.append(p[msk].mean()); ys.append(y[msk].mean())
    return xs, ys

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
for ax, (yy, pp, ttl) in zip(axes, [(yv, p_raw, "Uncalibrated"), (yv, p_iso, "Isotonic-recalibrated")]):
    xs, ys = reliability(yy, pp)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.plot(xs, ys, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_title(ttl, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "calibration_curve.png"), dpi=300)
plt.close(fig)
print("saved calibration_curve.png")

# ---- leakage sensitivity ----
if "leakage_sensitivity" in D:
    ls = D["leakage_sensitivity"]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    labels = ["Stratified 5-fold\n(dupes may cross folds)", "Grouped 5-fold\n(dupes isolated)"]
    vals = [ls["stratified_cv_auc"], ls["grouped_cv_auc"]]
    bars = ax.bar(labels, vals, color=["indianred", "steelblue"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.002, "%.4f" % v, ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.01, max(vals) + 0.008)
    ax.set_ylabel("Pooled OOF ROC-AUC")
    ax.set_title("Leakage sensitivity: duplicate-profile isolation")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "leakage_sensitivity.png"), dpi=300)
    plt.close(fig)
    print("saved leakage_sensitivity.png")

# ---- fairness by age with supports ----
if "fairness_diagnostics" in D:
    per = D["fairness_diagnostics"]["Age"]["fairness"]["per_group"]
    ages = sorted(per.keys(), key=lambda k: float(k))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(range(len(ages)), [per[a]["tpr"] for a in ages], color="steelblue", label="TPR (recall)")
    ax.plot(range(len(ages)), [per[a]["fpr"] for a in ages], "s--", color="firebrick", label="FPR", alpha=0.8)
    ax.set_xticks(range(len(ages)))
    ax.set_xticklabels(ages, fontsize=7)
    ax.set_xlabel("Age group (BRFSS code)")
    ax.set_ylabel("Rate")
    ax.set_title("Subgroup error rates by age (HGB, threshold 0.5)", fontsize=10)
    for i, a in enumerate(ages):
        ax.text(i, per[a]["tpr"] + 0.02, "n+=%d" % per[a]["n_pos"], ha="center", fontsize=6, rotation=90)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fairness_by_age.png"), dpi=300)
    plt.close(fig)
    print("saved fairness_by_age.png")

# ---- age gap under min support ----
if "age_robustness" in D:
    rb = D["age_robustness"]["by_min_support"]
    ks = sorted(rb.keys(), key=lambda x: int(x))
    fig, ax = plt.subplots(figsize=(5.4, 4))
    vals = [rb[k]["tpr_gap"] for k in ks]
    bars = ax.bar([str(k) for k in ks], vals, color="slateblue", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, "%.3f" % v, ha="center", fontsize=9)
    ax.set_xlabel("Minimum positives required in a band")
    ax.set_ylabel("Age TPR gap")
    ax.set_title("Age disparity under minimum-support constraint", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "age_gap_support.png"), dpi=300)
    plt.close(fig)
    print("saved age_gap_support.png")

# ---- feature selectors ----
if "feature_selectors" in D:
    fs = D["feature_selectors"]
    names = ["shap_k10", "mutual_info_k10", "correlation_k10"]
    if all(n in fs for n in names):
        vals = [fs[n]["auc"] for n in names]
        fig, ax = plt.subplots(figsize=(5.4, 4))
        bars = ax.bar(["SHAP", "Mutual info", "Correlation"], vals,
                      color=["steelblue", "indianred", "seagreen"], width=0.5)
        ax.axhline(fs["random_k10"]["auc_mean"], ls="--", color="gray",
                   label="Random 10-feature (%.4f)" % fs["random_k10"]["auc_mean"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.002, "%.4f" % v, ha="center", fontsize=9)
        ax.set_ylim(min(vals) - 0.01, max(vals) + 0.008)
        ax.set_ylabel("ROC-AUC (2015 test)")
        ax.set_title("Feature selector comparison (k=10, HGB)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "feature_selectors.png"), dpi=300)
        plt.close(fig)
        print("saved feature_selectors.png")

# ---- wave drift ----
if "wave_drift" in D:
    wd = D["wave_drift"]
    yrs = sorted(wd.keys())
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(yrs, [wd[y]["roc_auc"] for y in yrs], "o-", color="steelblue", label="ROC-AUC")
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("BRFSS test wave (model trained on 2015)")
    ax2 = ax.twinx()
    ax2.plot(yrs, [wd[y]["brier"] for y in yrs], "s--", color="firebrick", label="Brier")
    ax2.set_ylabel("Brier score")
    ax.set_title("Out-of-time discrimination and calibration drift", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "wave_drift.png"), dpi=300)
    plt.close(fig)
    print("saved wave_drift.png")

print("all audit figures written")