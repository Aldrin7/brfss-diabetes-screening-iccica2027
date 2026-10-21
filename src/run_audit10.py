import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results10.json")
GROUPS = ["Sex", "Age", "Income", "Education"]


def load_year(year):
    if year == 2015:
        return pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return pd.read_csv(os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % year))


def hgb(seed=42, balanced=True, **kw):
    params = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    if balanced:
        params["class_weight"] = "balanced"
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def ece_mce(y, p, n_bins=10, weights=None):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0; mce = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        conf = p[m].mean(); acc = y[m].mean()
        w = (m.sum() / len(y)) if weights is None else float(weights[m].sum() / weights.sum())
        ece += w * abs(acc - conf)
        mce = max(mce, abs(acc - conf))
    return float(ece), float(mce)


def ece_equal_mass(y, p, n_bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    ece = 0.0
    for ch in chunks:
        if len(ch) == 0: continue
        ece += (len(ch) / len(y)) * abs(y[ch].mean() - p[ch].mean())
    return float(ece)


def ece_adaptive(y, p, n_bins=10):
    """Equal-mass bins on the pooled score distribution (adaptive to score spread)."""
    return ece_equal_mass(y, p, n_bins)


def ece_threshold_weighted(y, p, thr=0.5):
    """Weight bins by the probability mass they carry in the positive region."""
    w = np.where(p >= thr, 2.0 * p, 2.0 * (1 - p))
    e, _ = ece_mce(y, p, n_bins=10, weights=w)
    return e


def group_stats(y, pred, groups, min_npos=20):
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        npos = int((y[m] == 1).sum()); nneg = int((y[m] == 0).sum())
        if npos < min_npos: continue
        per[g] = {"n": int(m.sum()), "n_pos": npos,
                  "tpr": float(((pred[m] == 1) & (y[m] == 1)).sum() / npos),
                  "fpr": float(((pred[m] == 1) & (y[m] == 0)).sum() / max(nneg, 1))}
    if not per:
        return {}, float("nan"), float("nan")
    tprs = [v["tpr"] for v in per.values()]
    fprs = [v["fpr"] for v in per.values()]
    return per, float(max(tprs) - min(tprs)), float(max(fprs) - min(fprs))


def beta_cal(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.log(p / (1 - p)), -np.log(1 - p)])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    return lr


def fit_safe(model, X, y, retries=4):
    """Fit on a contiguous float32 matrix, retrying on transient allocation errors."""
    import gc
    if isinstance(X, pd.DataFrame):
        Xm = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    else:
        Xm = np.ascontiguousarray(X, dtype=np.float32)
    ym = np.asarray(y)
    last = None
    for attempt in range(retries):
        try:
            model.fit(Xm, ym)
            return model
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            last = e
            gc.collect()
            time.sleep(1.0 + attempt)
    raise last


def predict_safe(model, X, chunk=40000):
    if isinstance(X, pd.DataFrame):
        Xm = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    else:
        Xm = np.ascontiguousarray(X, dtype=np.float32)
    out = np.empty(len(Xm), dtype=float)
    for i in range(0, len(Xm), chunk):
        out[i:i + chunk] = model.predict_proba(Xm[i:i + chunk])[:, 1]
    return out


def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
    yv = yte.values

    # ================= A. UNWEIGHTED vs CLASS-WEIGHTED =================
    print("=== A. Unweighted vs class-weighted training ===", flush=True)
    comp = {}
    for tag, balanced in [("class_weighted", True), ("unweighted", False)]:
        m = hgb(42, balanced=balanced)
        fit_safe(m, Xtr, ytr)
        pc = predict_safe(m, Xcal)
        pt = predict_safe(m, Xte)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc, ycal.values)
        b = beta_cal(pc, ycal.values)
        p_iso = np.clip(iso.predict(pt), 0, 1)
        Xb = np.column_stack([np.log(np.clip(pt, 1e-6, 1-1e-6) / (1 - np.clip(pt, 1e-6, 1-1e-6))),
                              -np.log(1 - np.clip(pt, 1e-6, 1 - 1e-6))])
        p_beta = b.predict_proba(Xb)[:, 1]
        # sensitivity-targeted threshold on calibrated scores (target 0.85)
        best_t = None
        for t in np.linspace(0.02, 0.98, 193):
            pr = (p_iso >= t).astype(int)
            s = ((pr == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
            if s >= 0.85:
                best_t = t
        pr = (p_iso >= best_t).astype(int)
        sens = float(((pr == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1))
        spec = float(((pr == 0) & (yv == 0)).sum() / max((yv == 0).sum(), 1))
        # net benefit at that threshold
        nb = None
        if best_t is not None and best_t < 1:
            tp = float(((pr == 1) & (yv == 1)).sum()); fp = float(((pr == 1) & (yv == 0)).sum())
            nb = tp / len(yv) - (fp / len(yv)) * (best_t / (1 - best_t))
        e_raw, m_raw = ece_mce(yv, pt)
        e_iso, m_iso = ece_mce(yv, p_iso)
        e_bet, m_bet = ece_mce(yv, p_beta)
        comp[tag] = {
            "roc_auc": float(roc_auc_score(yv, pt)),
            "auprc": float(average_precision_score(yv, pt)),
            "raw": {"brier": float(brier_score_loss(yv, pt)), "ece": e_raw, "mce": m_raw,
                    "mean_pred": float(pt.mean())},
            "isotonic": {"brier": float(brier_score_loss(yv, p_iso)), "ece": e_iso, "mce": m_iso,
                         "mean_pred": float(p_iso.mean())},
            "beta": {"brier": float(brier_score_loss(yv, p_beta)), "ece": e_bet, "mce": m_bet,
                     "mean_pred": float(p_beta.mean())},
            "sens_targeted_threshold": float(best_t) if best_t is not None else None,
            "sensitivity": sens, "specificity": spec, "net_benefit": nb,
        }
        print("  %-16s AUC=%.4f AUPRC=%.4f | raw ECE=%.4f -> iso %.4f / beta %.4f | "
              "sens-target thr=%.3f sens=%.3f spec=%.3f NB=%s" % (
            tag, comp[tag]["roc_auc"], comp[tag]["auprc"], e_raw, e_iso, e_bet,
            comp[tag]["sens_targeted_threshold"] or -1, sens, spec,
            ("%.4f" % nb) if nb is not None else "n/a"), flush=True)
    out["weighted_vs_unweighted"] = comp

    # ================= B. MULTI-CALIBRATION + GROUP THRESHOLDS =================
    print("\n=== B. Per-group (multi-)calibration before group-specific thresholds ===", flush=True)
    m = hgb(42, balanced=True); fit_safe(m, Xtr, ytr)
    p_cal = predict_safe(m, Xcal)
    p_te = predict_safe(m, Xte)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    p_te_iso = np.clip(iso.predict(p_te), 0, 1)

    def build_group_thresholds(prob_cal, prob_te, target=0.85):
        thr_map = {}
        for g in GROUPS:
            gc = Xcal[g].values; gv = ycal.values
            for lev in sorted(set(gc)):
                msk = gc == lev
                if (gv[msk] == 1).sum() < 20: continue
                bt = None
                for t in np.linspace(0.02, 0.98, 193):
                    pr = (prob_cal[msk] >= t).astype(int)
                    s = ((pr == 1) & (gv[msk] == 1)).sum() / max((gv[msk] == 1).sum(), 1)
                    if s >= target: bt = t
                if bt is not None: thr_map[(g, lev)] = bt
        pred = np.zeros(len(yv), dtype=int)
        for i in range(len(yv)):
            t = 0.5
            for g in GROUPS:
                key = (g, Xte[g].values[i])
                if key in thr_map: t = thr_map[key]; break
            pred[i] = 1 if prob_te[i] >= t else 0
        return pred

    p_cal_iso = np.clip(iso.predict(p_cal), 0, 1)
    # (i) global calibration then group thresholds
    pred_gc = build_group_thresholds(p_cal_iso, p_te_iso, 0.85)
    # (ii) per-group calibration then group thresholds
    p_te_pg = p_te.copy()
    p_cal_pg = p_cal.copy()
    for g in GROUPS:
        for lev in sorted(set(Xcal[g].values)):
            msk_c = Xcal[g].values == lev
            msk_t = Xte[g].values == lev
            if msk_c.sum() < 200 or (ycal.values[msk_c] == 1).sum() < 20: continue
            iso_g = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal[msk_c], ycal.values[msk_c])
            p_te_pg[msk_t] = np.clip(iso_g.predict(p_te[msk_t]), 0, 1)
            p_cal_pg[msk_c] = np.clip(iso_g.predict(p_cal[msk_c]), 0, 1)
    pred_pg = build_group_thresholds(p_cal_pg, p_te_pg, 0.85)
    mc = {}
    for tag, pred in [("global_calib_then_group_thr", pred_gc),
                      ("per_group_calib_then_group_thr", pred_pg)]:
        e = {"overall_sensitivity": float(((pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)),
             "overall_specificity": float(((pred == 0) & (yv == 0)).sum() / max((yv == 0).sum(), 1))}
        for g in GROUPS:
            _, tg, fg = group_stats(yv, pred, Xte[g].values)
            e[g] = {"tpr_gap": tg, "fpr_gap": fg}
        mc[tag] = e
        print("  %-34s sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f" % (
            tag, e["overall_sensitivity"], e["overall_specificity"], e["Sex"]["tpr_gap"],
            e["Age"]["tpr_gap"], e["Income"]["tpr_gap"], e["Education"]["tpr_gap"]), flush=True)
    out["multi_calibration"] = mc
    save_json(out, OUTPATH)

    # ================= C. LABEL-NOISE SENSITIVITY =================
    print("\n=== C. Label-noise sensitivity (1,963 conflicting rows) ===", flush=True)
    lab_all = df[TARGET].astype(int)
    first_lab = lab_all.groupby([df[c] for c in FEATURES], sort=False).transform("first")
    conflict_idx = np.where(df.duplicated(subset=FEATURES, keep="first").values &
                             (lab_all.values != first_lab.values))[0]
    conflict_keys = set(map(tuple, df.iloc[conflict_idx][FEATURES].to_numpy()))
    keep_mask = np.array([tuple(r) not in conflict_keys for r in df[FEATURES].to_numpy()])
    df_clean = df[keep_mask].reset_index(drop=True)
    lab_clean = df_clean[TARGET].astype(int)
    Xr2, Xt2, yr2, yt2 = train_test_split(df_clean[FEATURES], lab_clean, test_size=0.20, random_state=42, stratify=lab_clean)
    Xr2 = Xr2.reset_index(drop=True); yr2 = yr2.reset_index(drop=True)
    Xt2 = Xt2.reset_index(drop=True); yt2 = yt2.reset_index(drop=True)
    m2 = hgb(42); fit_safe(m2, Xr2, yr2)
    pt2 = predict_safe(m2, Xt2)
    e2, m2c = ece_mce(yt2.values, pt2)
    pred2 = (pt2 >= 0.5).astype(int)
    gaps2 = {}
    for g in GROUPS:
        _, tg, fg = group_stats(yt2.values, pred2, Xt2[g].values)
        gaps2[g] = {"tpr_gap": tg, "fpr_gap": fg}
    out["label_noise"] = {
        "n_conflicting_rows": int(len(conflict_idx)),
        "n_rows_removed": int((~keep_mask).sum()),
        "clean_auc": float(roc_auc_score(yt2.values, pt2)),
        "clean_auprc": float(average_precision_score(yt2.values, pt2)),
        "clean_ece_raw": e2, "clean_mce_raw": m2c,
        "clean_brier_raw": float(brier_score_loss(yt2.values, pt2)),
        "clean_gaps": gaps2,
        "reference_auc": float(roc_auc_score(yv, p_te)),
        "reference_ece_raw": ece_mce(yv, p_te)[0],
    }
    print("  removed %d rows -> AUC %.4f (ref %.4f, delta %+.4f) | ECE raw %.4f (ref %.4f)" % (
        out["label_noise"]["n_rows_removed"], out["label_noise"]["clean_auc"],
        out["label_noise"]["reference_auc"],
        out["label_noise"]["clean_auc"] - out["label_noise"]["reference_auc"],
        e2, out["label_noise"]["reference_ece_raw"]), flush=True)
    for g in GROUPS:
        print("    %-10s clean TPR gap %.4f (ref %.4f)" % (
            g, gaps2[g]["tpr_gap"], group_stats(yv, (p_te >= 0.5).astype(int), Xte[g].values)[1]), flush=True)
    save_json(out, OUTPATH)

    # ================= D. ECE BINNING ROBUSTNESS =================
    print("\n=== D. ECE binning robustness ===", flush=True)
    bin_res = {}
    for tag, fn in [("equal_width_10", lambda y, p: ece_mce(y, p, 10)[0]),
                    ("equal_width_15", lambda y, p: ece_mce(y, p, 15)[0]),
                    ("equal_width_20", lambda y, p: ece_mce(y, p, 20)[0]),
                    ("equal_mass_10", lambda y, p: ece_equal_mass(y, p, 10)),
                    ("equal_mass_20", lambda y, p: ece_equal_mass(y, p, 20)),
                    ("threshold_weighted", lambda y, p: ece_threshold_weighted(y, p, 0.5))]:
        bin_res[tag] = {"uncalibrated": fn(yv, p_te), "isotonic": fn(yv, p_te_iso)}
        print("  %-20s uncal=%.4f  isotonic=%.4f" % (tag, bin_res[tag]["uncalibrated"], bin_res[tag]["isotonic"]), flush=True)
    out["ece_binning_robustness"] = bin_res
    save_json(out, OUTPATH)

    # ================= E. OUT-OF-TIME STABILITY OF GROUP THRESHOLDS =================
    print("\n=== E. Out-of-time stability of group-specific thresholds (fit on 2015) ===", flush=True)
    thr_map = {}
    for g in GROUPS:
        gc = Xcal[g].values; gv = ycal.values
        for lev in sorted(set(gc)):
            msk = gc == lev
            if (gv[msk] == 1).sum() < 20: continue
            bt = None
            for t in np.linspace(0.02, 0.98, 193):
                pr = (p_cal_iso[msk] >= t).astype(int)
                s = ((pr == 1) & (gv[msk] == 1)).sum() / max((gv[msk] == 1).sum(), 1)
                if s >= 0.85: bt = t
            if bt is not None: thr_map[(g, lev)] = bt
    oot = {}
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr)
        Xw = wy[FEATURES].reset_index(drop=True); yw = wy[TARGET].astype(int).values
        pw = predict_safe(m, Xw)
        pw_iso = np.clip(iso.predict(pw), 0, 1)
        pred_w = np.zeros(len(yw), dtype=int)
        for i in range(len(yw)):
            t = 0.5
            for g in GROUPS:
                key = (g, Xw[g].values[i])
                if key in thr_map: t = thr_map[key]; break
            pred_w[i] = 1 if pw_iso[i] >= t else 0
        e = {"overall_sensitivity": float(((pred_w == 1) & (yw == 1)).sum() / max((yw == 1).sum(), 1)),
             "overall_specificity": float(((pred_w == 0) & (yw == 0)).sum() / max((yw == 0).sum(), 1))}
        for g in GROUPS:
            _, tg, fg = group_stats(yw, pred_w, Xw[g].values)
            e[g] = {"tpr_gap": tg, "fpr_gap": fg}
        # baseline global-threshold comparison
        pred_b = (pw_iso >= 0.5).astype(int)
        eb = {}
        for g in GROUPS:
            _, tg, fg = group_stats(yw, pred_b, Xw[g].values)
            eb[g] = {"tpr_gap": tg, "fpr_gap": fg}
        oot[str(yr)] = {"group_specific": e, "global_threshold": eb,
                        "auc": float(roc_auc_score(yw, pw))}
        print("  %d: group-specific sens=%.3f spec=%.3f | sex %.4f age %.4f inc %.4f edu %.4f "
              "|| global: sex %.4f age %.4f inc %.4f edu %.4f" % (
            yr, e["overall_sensitivity"], e["overall_specificity"], e["Sex"]["tpr_gap"],
            e["Age"]["tpr_gap"], e["Income"]["tpr_gap"], e["Education"]["tpr_gap"],
            eb["Sex"]["tpr_gap"], eb["Age"]["tpr_gap"], eb["Income"]["tpr_gap"], eb["Education"]["tpr_gap"]), flush=True)
    out["group_thresholds_out_of_time"] = oot
    save_json(out, OUTPATH)

    # ================= F. INTERSECTIONAL FAIRNESS =================
    print("\n=== F. Intersectional fairness (Sex x Age, Sex x Income, Age x Income) ===", flush=True)
    inter = {}
    pred_g = (p_te >= 0.5).astype(int)
    for a, b in [("Sex", "Age"), ("Sex", "Income"), ("Age", "Income")]:
        key = Xte[a].astype(str) + "|" + Xte[b].astype(str)
        per, tg, fg = group_stats(yv, pred_g, key.values, min_npos=30)
        inter["%s x %s" % (a, b)] = {"tpr_gap": tg, "fpr_gap": fg, "n_cells": len(per),
                                      "min_cell_n": min((v["n"] for v in per.values()), default=0),
                                      "min_cell_n_pos": min((v["n_pos"] for v in per.values()), default=0),
                                      "cells": per}
        print("  %-16s cells=%2d TPR gap=%.4f FPR gap=%.4f (min cell n=%d, min cell n_pos=%d)" % (
            a + " x " + b, len(per), tg, fg, inter["%s x %s" % (a, b)]["min_cell_n"],
            inter["%s x %s" % (a, b)]["min_cell_n_pos"]), flush=True)
    out["intersectional_fairness"] = inter
    save_json(out, OUTPATH)

    # ================= G. COUNTERFACTUAL FLIPS NEAR THE THRESHOLD =================
    print("\n=== G. Single-feature counterfactual flips near the decision threshold ===", flush=True)
    thr = 0.5
    near = np.where(np.abs(p_te - thr) < 0.05)[0]
    flips = {}
    for col in ["GenHlth", "HighBP", "BMI", "Age", "Income", "HighChol"]:
        changed = 0
        deltas = []
        for i in near[:20000]:
            for delta in ([-1, 1] if col in ("GenHlth", "Age") else [-5, 5] if col == "BMI" else [0, 1]):
                Xc = Xte.iloc[[i]].copy()
                v = float(Xc[col].iloc[0]) + delta
                Xc[col] = np.clip(v, float(Xte[col].min()), float(Xte[col].max()))
                if float(Xc[col].iloc[0]) == float(Xte[col].iloc[0]): continue
                pn = float(predict_safe(m, Xc)[0])
                if (p_te[i] >= thr) != (pn >= thr):
                    changed += 1
                    deltas.append(pn - p_te[i])
                    break
        flips[col] = {"n_near": int(len(near[:20000])), "n_flipped": int(changed),
                      "flip_rate": float(changed / max(len(near[:20000]), 1)),
                      "mean_prob_change_when_flipped": float(np.mean(deltas)) if deltas else float("nan")}
        print("  %-9s near-threshold n=%d, single-feature flips=%d (%.2f%%)" % (
            col, flips[col]["n_near"], flips[col]["n_flipped"], 100 * flips[col]["flip_rate"]), flush=True)
    out["counterfactual_flips"] = flips

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
