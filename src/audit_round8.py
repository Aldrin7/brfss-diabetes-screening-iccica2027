"""Round-8 review computations: per-group discrimination and calibration (2015 and 2021),
age-band supports, subgroup net benefit, and per-group calibration transport.

Answers reviewer questions 4, 5, 6 and 7 of round 8.

Protocol (identical to the manuscript):
  * 2015 data, stratified 80/20 split (seed 42) -> train/test
  * inside train: 64/16/20 train/calibration/test
  * model: HistGradientBoostingClassifier(max_iter=150, lr=0.1, max_leaf_nodes=31,
            class_weight="balanced", random_state=42)
  * global isotonic calibrator fitted on the 2015 calibration split
  * recommended operating point: global sensitivity-targeted threshold 0.115 on
    recalibrated scores (transportable to later waves)

Outputs results/logs/audit_results14.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FEATURES, TARGET, load_raw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
P2015 = os.path.join(ROOT, "data", "raw", "brfss_multi", "cdc_brfss_diabetes_2015.csv")
P2021 = os.path.join(ROOT, "data", "raw", "brfss_multi", "cdc_brfss_diabetes_2021.csv")
THR = 0.115          # recommended global sensitivity-targeted threshold
PT = 0.10            # threshold probability used in the decision-curve analysis
MIN_SUPPORT = 500    # positives, support restriction for the age estimate

# BRFSS age categories (the 13 bands used in the manuscript)
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
    8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def subgroup_table(df, y, p, thr, groups):
    out = {}
    for g in sorted(df[groups].dropna().unique()):
        m = (df[groups] == g).values
        if m.sum() < 50 or y[m].sum() < 10:
            continue
        yt, pt_ = y[m], p[m]
        pred = (pt_ >= thr).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        n = int(m.sum())
        npos = int(yt.sum())
        nb = tpr * yt.mean() - fpr * (1 - yt.mean()) * (PT / (1 - PT))
        try:
            auc = float(roc_auc_score(yt, pt_))
        except Exception:
            auc = None
        out[str(g)] = {
            "n": n, "n_pos": npos, "prevalence": float(yt.mean()),
            "roc_auc": auc, "ece_10bin": ece(yt, pt_), "brier": float(brier_score_loss(yt, pt_)),
            "tpr": float(tpr), "fpr": float(fpr), "net_benefit_pt0.10": float(nb),
        }
    return out


def main():
    t0 = time.time()
    df = load_raw(P2015)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).values
    # 64/16/20 train/calibration/test split, identical to run_audit3.py and the manuscript
    Xrest, X_te, yrest, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_cal, y_tr, y_cal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    print("split:", len(y_tr), len(y_cal), len(y_te), flush=True)

    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(X_cal)[:, 1], y_cal)
    p_te = iso.predict(model.predict_proba(X_te)[:, 1])
    df_te = df.loc[X_te.index]
    print("2015 test done", flush=True)

    res = {"protocol": {"threshold": THR, "pt": PT, "min_support_positives": MIN_SUPPORT,
                        "model": "HistGradientBoosting(max_iter=150, lr=0.1, leaves=31, class_weight=balanced, seed=42)",
                        "calibrator": "isotonic fitted on the 2015 calibration split (16%)"}}

    res["2015_test_overall"] = {
        "n": int(len(y_te)), "n_pos": int(y_te.sum()), "prevalence": float(y_te.mean()),
        "roc_auc_raw": float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])),
        "roc_auc_recalibrated": float(roc_auc_score(y_te, p_te)),
        "ece_10bin_recalibrated": ece(y_te, p_te),
        "brier_recalibrated": float(brier_score_loss(y_te, p_te)),
    }
    # age-band definitions and positive counts on the FULL 2015 data (reviewer Q5)
    full = df.groupby("Age")[TARGET].agg(["size", "sum"])
    res["age_band_definitions_full_2015"] = {
        str(int(b)): {"label": AGE_LABELS.get(int(b), "band %d" % int(b)),
                      "n": int(r["size"]), "n_pos": int(r["sum"]),
                      "retained_at_min_support_%d" % MIN_SUPPORT: bool(r["sum"] >= MIN_SUPPORT)}
        for b, r in full.iterrows()
    }
    with open(os.path.join(LOGS, "age_band_supports.csv"), "w", encoding="utf-8") as fh:
        fh.write("age_code,label,n,n_pos,retained_at_min_support_%d\n" % MIN_SUPPORT)
        for b, r in full.iterrows():
            fh.write("%d,%s,%d,%d,%d\n" % (int(b), AGE_LABELS.get(int(b), ""), int(r["size"]),
                                           int(r["sum"]), int(r["sum"] >= MIN_SUPPORT)))

    res["2015_test_subgroups"] = {
        g: subgroup_table(df_te, y_te, p_te, THR, g) for g in ("Age", "Sex", "Income", "Education")
    }
    res["2015_age_bands"] = {
        str(b): {"n": int((df_te["Age"] == b).sum()),
                 "n_pos": int(y_te[(df_te["Age"] == b).values].sum())}
        for b in sorted(df_te["Age"].unique())
    }

    # ---- 2021: transport the global calibrator and the threshold -------------
    d21 = load_raw(P2021)
    X21 = d21[FEATURES].copy()
    y21 = d21[TARGET].astype(int).values
    p21 = iso.predict(model.predict_proba(X21)[:, 1])
    res["2021_subgroups"] = {
        g: subgroup_table(d21, y21, p21, THR, g) for g in ("Age", "Sex", "Income", "Education")
    }
    res["2021_overall"] = {
        "n": int(len(y21)), "n_pos": int(y21.sum()), "prevalence": float(y21.mean()),
        "roc_auc": float(roc_auc_score(y21, p21)), "ece_10bin": ece(y21, p21),
        "brier": float(brier_score_loss(y21, p21)),
    }
    print("2021 done", flush=True)

    # ---- per-group (age) recalibration transport, Q6 --------------------------
    def group_iso_fit_predict(groups_cal, groups_apply):
        """Per-group (age band) isotonic fitted on the 2015 calibration split, applied to 2021."""
        p = np.zeros(len(y21))
        p_cal = model.predict_proba(X_cal)[:, 1]
        for b in sorted(set(groups_cal)):
            m_tr = (groups_cal == b)
            m_ap = (groups_apply == b)
            if m_tr.sum() < 200 or y_cal[m_tr].sum() < 20:
                p[m_ap] = iso.predict(model.predict_proba(X21[m_ap])[:, 1])
                continue
            g = IsotonicRegression(out_of_bounds="clip")
            g.fit(p_cal[m_tr], y_cal[m_tr])
            p[m_ap] = g.predict(model.predict_proba(X21[m_ap])[:, 1])
        return p

    m_cal = df.loc[X_cal.index, "Age"].values
    p21_group = group_iso_fit_predict(m_cal, d21["Age"].values)
    res["per_group_calibration_transport"] = {
        "global_transport_ece_age_bands": {
            b: ece(y21[(d21["Age"] == b).values], p21[(d21["Age"] == b).values])
            for b in sorted(d21["Age"].unique()) if ((d21["Age"] == b).values).sum() >= 50
        },
        "per_group_transport_ece_age_bands": {
            b: ece(y21[(d21["Age"] == b).values], p21_group[(d21["Age"] == b).values])
            for b in sorted(d21["Age"].unique()) if ((d21["Age"] == b).values).sum() >= 50
        },
        "overall_ece_global_transport": ece(y21, p21),
        "overall_ece_per_group_transport": ece(y21, p21_group),
    }
    print("per-group transport done", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    out = os.path.join(LOGS, "audit_results14.json")
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out, "elapsed", res["elapsed_s"], "s")

    # compact console summary for the manuscript sentences
    ag15 = res["2015_test_subgroups"]["Age"]
    ag21 = res["2021_subgroups"]["Age"]
    a15 = [v["roc_auc"] for v in ag15.values() if v["roc_auc"]]
    a21 = [v["roc_auc"] for v in ag21.values() if v["roc_auc"]]
    e15 = [v["ece_10bin"] for v in ag15.values()]
    e21 = [v["ece_10bin"] for v in ag21.values()]
    nb21 = [v["net_benefit_pt0.10"] for v in ag21.values()]
    print("AGE 2015: AUC %.4f-%.4f | ECE %.4f-%.4f" % (min(a15), max(a15), min(e15), max(e15)))
    print("AGE 2021: AUC %.4f-%.4f | ECE %.4f-%.4f | NB %.4f-%.4f" % (min(a21), max(a21), min(e21), max(e21), min(nb21), max(nb21)))
    bands = res["2015_age_bands"]
    keep = {k: v for k, v in bands.items() if v["n_pos"] >= MIN_SUPPORT}
    print("bands kept (>=%d pos): %d of %d -> %s" % (MIN_SUPPORT, len(keep), len(bands), sorted(keep)))
    print(json.dumps(keep, indent=1))
    print("2021 overall:", json.dumps(res["2021_overall"]))
    print("transport ECE global %.4f vs per-group %.4f" % (
        res["per_group_calibration_transport"]["overall_ece_global_transport"],
        res["per_group_calibration_transport"]["overall_ece_per_group_transport"]))


if __name__ == "__main__":
    main()
