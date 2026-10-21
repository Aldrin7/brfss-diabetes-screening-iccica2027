import os, sys, json, time, functools, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results7.json")
GROUPS = ["Sex", "Age", "Income", "Education"]


def hgb(seed=42, **kw):
    params = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                  class_weight="balanced", random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


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
    p_cal = m.predict_proba(Xcal)[:, 1]
    p_te = m.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_te_cal = np.clip(iso.predict(p_te), 0, 1)

    # ======= I. MULTI-GROUP CONSTRAINED THRESHOLD OPTIMIZATION =======
    print("=== I. Constrained threshold: min FN s.t. bounded TPR gap across ALL stratifiers ===", flush=True)
    grid = []
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p_te >= t).astype(int)
        fn = int(((pred == 0) & (yv == 1)).sum())
        gaps = {}
        for g in GROUPS:
            _, tg, fg = group_stats(yv, pred, Xte[g].values)
            gaps[g] = {"tpr_gap": tg, "fpr_gap": fg}
        worst_tpr = max(v["tpr_gap"] for v in gaps.values())
        worst_fpr = max(v["fpr_gap"] for v in gaps.values())
        grid.append({"t": float(t), "fn": fn, "worst_tpr_gap": worst_tpr,
                     "worst_fpr_gap": worst_fpr, "gaps": gaps})
    multi = {}
    for delta in [0.10, 0.20, 0.30, 0.50, 1.01]:
        feas = [g for g in grid if g["worst_tpr_gap"] <= delta]
        if feas:
            best = min(feas, key=lambda g: g["fn"])
            multi["%.2f" % delta] = {"feasible": True, "threshold": best["t"], "fn": best["fn"],
                                     "achieved_worst_tpr_gap": best["worst_tpr_gap"],
                                     "achieved_worst_fpr_gap": best["worst_fpr_gap"],
                                     "per_group": best["gaps"]}
        else:
            multi["%.2f" % delta] = {"feasible": False}
        b = multi["%.2f" % delta]
        print("  max TPR gap <= %.2f : %s" % (delta, {k: v for k, v in b.items() if k != "per_group"}), flush=True)
    out["multi_group_constrained_threshold"] = {"by_delta": multi}

    # ======= J. GROUP-SPECIFIC THRESHOLDS (fairness mitigation) =======
    print("\n=== J. Group-specific thresholds targeting 0.85 sensitivity per group ===", flush=True)
    for target in [0.80, 0.85, 0.90]:
        thr_map = {}
        for g in GROUPS:
            gc = Xcal[g].values; gv = ycal.values
            for lev in sorted(set(gc)):
                msk = gc == lev
                if (gv[msk] == 1).sum() < 20: continue
                best_t = None
                for t in np.linspace(0.02, 0.98, 193):
                    pr = (p_cal[msk] >= t).astype(int)
                    s = ((pr == 1) & (gv[msk] == 1)).sum() / max((gv[msk] == 1).sum(), 1)
                    if s >= target:
                        best_t = t
                if best_t is not None:
                    thr_map[(g, lev)] = best_t
        # apply group-specific thresholds on test
        pred_gs = np.zeros(len(yv), dtype=int)
        for i in range(len(yv)):
            t = 0.5
            for g in GROUPS:
                key = (g, Xte[g].values[i])
                if key in thr_map:
                    t = thr_map[key]; break
            pred_gs[i] = 1 if p_te[i] >= t else 0
        pred_gl = (p_te >= 0.5).astype(int)
        res = {"target_sensitivity": target, "n_group_thresholds": len(thr_map),
               "global": {}, "group_specific": {}}
        for g in GROUPS:
            _, tg_g, fg_g = group_stats(yv, pred_gl, Xte[g].values)
            _, tg_s, fg_s = group_stats(yv, pred_gs, Xte[g].values)
            res["global"][g] = {"tpr_gap": tg_g, "fpr_gap": fg_g}
            res["group_specific"][g] = {"tpr_gap": tg_s, "fpr_gap": fg_s}
        res["global"]["overall_sensitivity"] = float(((pred_gl == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1))
        res["global"]["overall_specificity"] = float(((pred_gl == 0) & (yv == 0)).sum() / max((yv == 0).sum(), 1))
        res["group_specific"]["overall_sensitivity"] = float(((pred_gs == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1))
        res["group_specific"]["overall_specificity"] = float(((pred_gs == 0) & (yv == 0)).sum() / max((yv == 0).sum(), 1))
        out.setdefault("group_specific_thresholds", {})["%.2f" % target] = res
        print("  target %.2f: global sens=%.3f spec=%.3f | group-specific sens=%.3f spec=%.3f" % (
            target, res["global"]["overall_sensitivity"], res["global"]["overall_specificity"],
            res["group_specific"]["overall_sensitivity"], res["group_specific"]["overall_specificity"]), flush=True)
        for g in GROUPS:
            print("     %-10s TPR gap %.4f -> %.4f | FPR gap %.4f -> %.4f" % (
                g, res["global"][g]["tpr_gap"], res["group_specific"][g]["tpr_gap"],
                res["global"][g]["fpr_gap"], res["group_specific"][g]["fpr_gap"]), flush=True)
    save_json(out, OUTPATH)

    # ======= K. SCENARIO PROBES: subgroup-specific + calibrated =======
    print("\n=== K. Scenario probes: subgroup-specific and on calibrated scores ===", flush=True)
    scen = {}
    defs = {
        "BMI +5": ("BMI", lambda v: v + 5),
        "GenHlth +1": ("GenHlth", lambda v: v + 1),
    }
    for label, (col, fn) in defs.items():
        Xp = Xte.copy()
        lo, hi = float(Xp[col].min()), float(Xp[col].max())
        Xp[col] = np.clip(fn(Xp[col].astype(float)), lo, hi)
        p_raw_s = m.predict_proba(Xp)[:, 1]
        p_raw_b = p_te
        d_raw = p_raw_s - p_raw_b
        # calibrated
        c_raw = np.clip(iso.predict(p_raw_b), 0, 1)
        c_new = np.clip(iso.predict(p_raw_s), 0, 1)
        d_cal = c_new - c_raw
        entry = {"column": col, "feature_range": [lo, hi],
                 "raw": {"mean_change": float(d_raw.mean()),
                         "share_increased": float((d_raw > 0).mean())},
                 "calibrated": {"mean_change": float(d_cal.mean()),
                                "share_increased": float((d_cal > 0).mean())}}
        for g in ["Sex", "Age", "Income"]:
            sub = {}
            for lev in sorted(set(Xte[g].values)):
                msk = Xte[g].values == lev
                if msk.sum() < 500: continue
                sub[str(lev)] = {"n": int(msk.sum()),
                                 "raw_mean_change": float(d_raw[msk].mean()),
                                 "cal_mean_change": float(d_cal[msk].mean())}
            entry["by_" + g] = sub
        scen[label] = entry
        print("  %-12s raw mean %+.4f | calibrated mean %+.4f" % (
            label, entry["raw"]["mean_change"], entry["calibrated"]["mean_change"]), flush=True)
    out["scenario_subgroup"] = scen
    save_json(out, OUTPATH)

    # ======= L. REPRODUCIBILITY MANIFEST (SHA256) =======
    print("\n=== L. Reproducibility manifest ===", flush=True)
    man = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".venv", ".git", "__pycache__", "raw", "brfss_multi")]
        for fn in filenames:
            if fn.endswith((".py", ".json", ".tex", ".md", ".txt", ".png", ".pdf")):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT)
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                man[rel] = {"sha256": h, "bytes": os.path.getsize(p)}
    out["manifest"] = {"n_files": len(man), "files": man}
    print("  hashed %d files" % len(man), flush=True)
    save_json(out, OUTPATH)
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
