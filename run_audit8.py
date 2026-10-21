import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results8.json")
GROUPS = ["Sex", "Age", "Income", "Education"]


def hgb(seed=42, **kw):
    params = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                  class_weight="balanced", random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def group_stats(y, pred, groups, min_npos=20):
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        npos = int((y[m] == 1).sum()); nneg = int((y[m] == 0).sum())
        if npos < min_npos: continue
        per[g] = {"n": int(m.sum()), "n_pos": npos,
                  "tpr": float(((pred[m] == 1) & (y[m] == 1)).sum() / npos),
                  "fpr": float(((pred[m] == 1) & (y[m] == 0)).sum() / max(nneg, 1)),
                  "sel_rate": float(pred[m].mean())}
    if not per:
        return {}, float("nan"), float("nan")
    tprs = [v["tpr"] for v in per.values()]
    fprs = [v["fpr"] for v in per.values()]
    return per, float(max(tprs) - min(tprs)), float(max(fprs) - min(fprs))


def main():
    t0 = time.time()
    out = {}
    df = pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yv = yte.values

    m = hgb(42); m.fit(Xtr, ytr)
    p_te = m.predict_proba(Xte)[:, 1]

    # ======= M. NON-DEGENERATE CONSTRAINED THRESHOLD SELECTION =======
    # Maximise specificity (minimise FP) subject to:
    #   (a) overall sensitivity >= s_target, and
    #   (b) worst-group TPR gap (over Sex/Age/Income/Education) <= delta
    print("=== M. Constrained threshold: max specificity s.t. sensitivity>=s AND worst TPR gap<=delta ===", flush=True)
    rows = []
    for t in np.linspace(0.02, 0.98, 193):
        pred = (p_te >= t).astype(int)
        tp = int(((pred == 1) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum()); tn = int(((pred == 0) & (yv == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        gaps = {}
        for g in GROUPS:
            _, tg, fg = group_stats(yv, pred, Xte[g].values)
            gaps[g] = {"tpr_gap": tg, "fpr_gap": fg}
        rows.append({"t": float(t), "sens": sens, "spec": spec, "fp": fp, "fn": fn,
                     "worst_tpr_gap": max(v["tpr_gap"] for v in gaps.values()),
                     "worst_fpr_gap": max(v["fpr_gap"] for v in gaps.values()),
                     "gaps": gaps})
    sel = {}
    for s_t in [0.80, 0.85, 0.90]:
        for delta in [0.05, 0.10, 0.20, 0.30, 0.50]:
            feas = [r for r in rows if r["sens"] >= s_t and r["worst_tpr_gap"] <= delta]
            if feas:
                b = max(feas, key=lambda r: r["spec"])
                sel["s%.2f_d%.2f" % (s_t, delta)] = {
                    "feasible": True, "threshold": b["t"], "sens": b["sens"], "spec": b["spec"],
                    "fn": b["fn"], "fp": b["fp"],
                    "achieved_worst_tpr_gap": b["worst_tpr_gap"],
                    "achieved_worst_fpr_gap": b["worst_fpr_gap"],
                    "per_group": b["gaps"]}
            else:
                sel["s%.2f_d%.2f" % (s_t, delta)] = {"feasible": False}
    out["constrained_threshold_nondegenerate"] = sel
    print("  sensitivity target 0.85:", flush=True)
    for delta in [0.05, 0.10, 0.20, 0.30, 0.50]:
        k = "s0.85_d%.2f" % delta
        b = sel[k]
        if b.get("feasible"):
            print("    gap<=%.2f -> thr=%.3f sens=%.3f spec=%.3f (worst TPR gap %.3f, worst FPR gap %.3f)" % (
                delta, b["threshold"], b["sens"], b["spec"], b["achieved_worst_tpr_gap"], b["achieved_worst_fpr_gap"]), flush=True)
        else:
            print("    gap<=%.2f -> INFEASIBLE" % delta, flush=True)
    # baseline reference
    base = [r for r in rows if abs(r["t"] - 0.5) < 0.006][0]
    print("  baseline t=0.50: sens=%.3f spec=%.3f worst TPR gap=%.3f" % (
        base["sens"], base["spec"], base["worst_tpr_gap"]), flush=True)
    out["constrained_threshold_baseline"] = base
    save_json(out, OUTPATH)

    # ======= N. SEVERITY-STRATIFIED DISCRIMINATION AND CALIBRATION =======
    print("\n=== N. Severity-stratified discrimination (all rows within stratum) ===", flush=True)
    from sklearn.isotonic import IsotonicRegression
    p_cal = m.predict_proba(Xcal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_cal_te = np.clip(iso.predict(p_te), 0, 1)
    gh = Xte["GenHlth"].values
    bmi = Xte["BMI"].values
    strata = {
        "GenHlth 1-2 (good/very good)": gh <= 2,
        "GenHlth 3 (fair)": gh == 3,
        "GenHlth 4-5 (poor/very poor)": gh >= 4,
        "BMI < 30": bmi < 30,
        "BMI >= 30": bmi >= 30,
    }
    sev = {}
    for name, msk in strata.items():
        ys = yv[msk]
        if len(np.unique(ys)) < 2:
            continue
        ps = p_te[msk]
        pcs = p_cal_te[msk]
        e, mc = 0.0, 0.0
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pcs, edges) - 1, 0, 9)
        for b in range(10):
            bmask = idx == b
            if bmask.sum() == 0: continue
            e += (bmask.sum() / len(ys)) * abs(ys[bmask].mean() - pcs[bmask].mean())
            mc = max(mc, abs(ys[bmask].mean() - pcs[bmask].mean()))
        sev[name] = {"n": int(msk.sum()), "n_pos": int(ys.sum()),
                     "prevalence": float(ys.mean()),
                     "within_stratum_auc": float(roc_auc_score(ys, ps)),
                     "brier_raw": float(brier_score_loss(ys, ps)),
                     "brier_calibrated": float(brier_score_loss(ys, pcs)),
                     "ece_calibrated": float(e), "mce_calibrated": float(mc),
                     "mean_pred_raw": float(ps.mean()),
                     "mean_pred_calibrated": float(pcs.mean())}
        print("  %-30s n=%6d pos=%5d prev=%.3f AUC=%.4f brier raw->cal %.4f->%.4f ECE_cal=%.4f" % (
            name, sev[name]["n"], sev[name]["n_pos"], sev[name]["prevalence"],
            sev[name]["within_stratum_auc"], sev[name]["brier_raw"],
            sev[name]["brier_calibrated"], sev[name]["ece_calibrated"]), flush=True)
    out["severity_stratified"] = sev
    save_json(out, OUTPATH)
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
