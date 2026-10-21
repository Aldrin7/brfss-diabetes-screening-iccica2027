"""Column-sized figures for the two-column manuscript.

Produces
  assets/figures/decision_curve_col.png  decision-curve net benefit (utility contribution)
  assets/figures/age_band_diagnostics.png  per-age-band discrimination, calibration and net
                                          benefit from the round-8 audit (results/logs/
                                          audit_results14.json)

Both are sized for a 3.15 in IEEE column (>= 300 dpi at that size) and use 6 pt type.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
FIGS = os.path.join(ROOT, "assets", "figures")
os.makedirs(FIGS, exist_ok=True)

AGE_LABELS = {7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
              12: "75-79", 13: "80+"}

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6.5,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "axes.linewidth": 0.6, "figure.dpi": 300,
})


def load_all():
    d = {}
    for f in sorted(glob.glob(os.path.join(LOGS, "audit_results*.json"))):
        try:
            d.update(json.load(open(f)))
        except Exception:
            pass
    return d


def decision_curve():
    dc = load_all().get("decision_curve")
    if not dc:
        print("no decision_curve in the logs")
        return
    t = np.array(dc["thresholds"])
    fig, ax = plt.subplots(figsize=(3.2, 2.15))
    ax.plot(t, dc["model"], "-o", ms=2.2, color="steelblue", lw=1.1, label="HGB model")
    ax.plot(t, dc["treat_all"], "--s", ms=2.2, color="firebrick", lw=1.1, label="Treat all")
    ax.plot(t, dc["treat_none"], ":", color="gray", lw=1.0, label="Treat none")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0.10, color="0.6", lw=0.6, ls="-.", label="0.10")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.12, 0.11)
    ax.set_xlim(0.02, 0.55)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower left", frameon=True, handlelength=1.4, borderpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "decision_curve_col.png"), dpi=300)
    plt.close(fig)
    print("saved decision_curve_col.png")


def age_band_diagnostics():
    d = load_all()
    r14 = d.get("protocol")
    res = json.load(open(os.path.join(LOGS, "audit_results14.json")))
    a15 = res["2015_test_subgroups"]["Age"]
    a21 = res["2021_subgroups"]["Age"]
    bands = sorted([k for k, v in a15.items() if v["n_pos"] >= 500], key=float)
    x = np.arange(len(bands))
    lab = [AGE_LABELS.get(int(float(b)), b) for b in bands]

    auc15 = [a15[b]["roc_auc"] for b in bands]
    auc21 = [a21[b]["roc_auc"] for b in bands]
    ece15 = [a15[b]["ece_10bin"] for b in bands]
    ece21 = [a21[b]["ece_10bin"] for b in bands]
    nb15 = [a15[b]["net_benefit_pt0.10"] for b in bands]
    nb21 = [a21[b]["net_benefit_pt0.10"] for b in bands]

    fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.45))
    a, b, c = axes

    a.plot(x, auc15, "-o", ms=2.4, color="steelblue", lw=1.0, label="2015")
    a.plot(x, auc21, "--s", ms=2.4, color="firebrick", lw=1.0, label="2021")
    a.set_xticks(x)
    a.set_xticklabels(lab, rotation=90)
    a.set_ylabel("ROC-AUC")
    a.set_ylim(0.65, 0.9)
    a.grid(alpha=0.3, lw=0.4)
    a.legend(loc="lower left", frameon=False, handlelength=1.3)
    a.set_title("(a) discrimination", pad=2)

    b.plot(x, ece15, "-o", ms=2.4, color="steelblue", lw=1.0, label="2015")
    b.plot(x, ece21, "--s", ms=2.4, color="firebrick", lw=1.0, label="2021")
    b.set_xticks(x)
    b.set_xticklabels([])
    b.set_ylabel("ECE (10 bins)")
    b.set_ylim(0, 0.03)
    b.grid(alpha=0.3, lw=0.4)
    b.set_title("(b) calibration", pad=2)

    c.plot(x, nb15, "-o", ms=2.4, color="steelblue", lw=1.0, label="2015")
    c.plot(x, nb21, "--s", ms=2.4, color="firebrick", lw=1.0, label="2021")
    c.axhline(0, color="black", lw=0.5)
    c.set_xticks(x)
    c.set_xticklabels(lab, rotation=90)
    c.set_ylabel("Net benefit ($p_t$=0.10)")
    c.grid(alpha=0.3, lw=0.4)
    c.set_title("(c) net benefit", pad=2)

    fig.tight_layout(pad=0.3, w_pad=0.9)
    fig.savefig(os.path.join(FIGS, "age_band_diagnostics.png"), dpi=300)
    plt.close(fig)
    print("saved age_band_diagnostics.png (bands:", ", ".join(lab), ")")




def calibration_reliability():
    """Reliability of the class-weighted model before and after isotonic recalibration."""
    import os
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split as tts

    raw_csv = os.path.join(ROOT, "data", "raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(raw_csv)
    FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                 "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                 "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                 "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
    TARGET = "Diabetes_binary"
    Xrest, Xte, yrest, yte = tts(df[FEATURES], df[TARGET].astype(int), test_size=0.20,
                                  random_state=42, stratify=df[TARGET].astype(int))
    Xtr, Xcal, ytr, ycal = tts(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                       class_weight="balanced", random_state=42)
    m.fit(Xtr.reset_index(drop=True), ytr.reset_index(drop=True))
    p_raw = m.predict_proba(Xte)[:, 1]
    yv = yte.astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(m.predict_proba(Xcal.reset_index(drop=True))[:, 1], ycal.reset_index(drop=True).values)
    p_iso = np.clip(iso.predict(p_raw), 0, 1)

    def reliability(y, pr, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(pr, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            msk = idx == b
            if msk.sum() < 30:
                continue
            xs.append(pr[msk].mean())
            ys.append(y[msk].mean())
        return xs, ys

    fig, ax = plt.subplots(figsize=(3.2, 2.05))
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect calibration")
    xs, ys = reliability(yv, p_raw)
    ax.plot(xs, ys, "o-", ms=2.6, color="firebrick", lw=1.1, label="Uncalibrated")
    xs, ys = reliability(yv, p_iso)
    ax.plot(xs, ys, "s-", ms=2.6, color="steelblue", lw=1.1, label="Isotonic")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", frameon=True, handlelength=1.4, borderpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "calibration_reliability_col.png"), dpi=300)
    plt.close(fig)
    print("saved calibration_reliability_col.png")


def threshold_transfer():
    """TPR gap of each policy's target attribute, in sample (2015) and out of time (2021)."""
    d = load_all()
    pol15 = d.get("policy_comparison_2015")
    oot = d.get("policy_out_of_time", {})
    if not pol15 or "2021" not in oot:
        print("no policy comparison in the logs")
        return
    out21 = oot["2021"]
    attrs = ["Sex", "Age", "Income", "Education"]
    # each policy is scored on its own target attribute
    in_sample = [pol15[a][a]["tpr_gap"] for a in attrs]
    out_time = [out21[a][a]["tpr_gap"] for a in attrs]
    x = np.arange(len(attrs))
    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    ax.bar(x - 0.19, in_sample, 0.36, color="steelblue", label="2015 (in sample)")
    ax.bar(x + 0.19, out_time, 0.36, color="firebrick", label="2021 (transported)")
    ax.set_xticks(x)
    ax.set_xticklabels(attrs)
    ax.set_ylabel("TPR gap of the target attribute")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, lw=0.4, axis="y")
    ax.legend(loc="upper left", frameon=True, handlelength=1.3, borderpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "threshold_transfer_col.png"), dpi=300)
    plt.close(fig)
    print("saved threshold_transfer_col.png")
    for a, i, o in zip(attrs, in_sample, out_time):
        print("   %-10s in-sample %.4f -> 2021 %.4f" % (a, i, o))


def shap_column():
    """Compact SHAP bar chart sized for an IEEE column."""
    shap_path = os.path.join(LOGS, "shap.json")
    if not os.path.exists(shap_path):
        print("no shap.json in the logs")
        return
    shap = json.load(open(shap_path))
    if "top_features" not in shap:
        print("no top_features in shap.json")
        return
    feats = shap["top_features"]
    names = [f[0] for f in feats][::-1]
    vals = [f[1] for f in feats][::-1]
    fig, ax = plt.subplots(figsize=(3.2, 2.35))
    ax.barh(range(len(names)), vals, color="steelblue", height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(alpha=0.3, lw=0.4, axis="x")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "shap_top_features_col.png"), dpi=300)
    plt.close(fig)
    print("saved shap_top_features_col.png")


if __name__ == "__main__":
    shap_column()
    decision_curve()
    age_band_diagnostics()
    calibration_reliability()
    threshold_transfer()
