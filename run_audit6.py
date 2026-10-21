import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results6.json")


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


def cal_block(y, p):
    e, mc = ece_mce(y, p)
    return {"n": int(len(y)), "brier": float(brier_score_loss(y, p)), "ece": e, "mce": mc,
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


def subgroup_cal(y, p, groups, min_n=200):
    per = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < min_n: continue
        per[str(g)] = cal_block(y[m], p[m])
    if not per:
        return {}
    return {"max_group_ece": float(max(v["ece"] for v in per.values())),
            "mean_group_ece": float(np.mean([v["ece"] for v in per.values()])),
            "max_group_brier": float(max(v["brier"] for v in per.values())),
            "n_groups": len(per)}


def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    lab = df[TARGET].astype(int)

    # 2015 train / calib / test
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)

    m = hgb(42); m.fit(Xtr, ytr)
    p_cal = m.predict_proba(Xcal)[:, 1]
    p_te = m.predict_proba(Xte)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)
    pl = LogisticRegression().fit(p_cal.reshape(-1, 1), ycal.values)

    print("=== Per-wave calibration transport (2015-fitted maps, NOT refit) ===", flush=True)
    waves = {}
    for yr in [2015, 2017, 2019, 2021]:
        if yr == 2015:
            Xw, yw = Xte, yte.reset_index(drop=True).values
        else:
            wy = load_year(yr)
            Xw = wy[FEATURES].reset_index(drop=True)
            yw = wy[TARGET].astype(int).values
        pw = m.predict_proba(Xw)[:, 1]
        entry = {"n": int(len(yw)), "prevalence": float(yw.mean()),
                 "roc_auc": float(roc_auc_score(yw, pw)),
                 "raw": cal_block(yw, pw),
                 "isotonic_transported": cal_block(yw, np.clip(iso.predict(pw), 0, 1)),
                 "platt_transported": cal_block(yw, pl.predict_proba(pw.reshape(-1, 1))[:, 1])}
        for g in ["Sex", "Age", "Income"]:
            entry["subgroup_" + g] = {
                "raw": subgroup_cal(yw, pw, Xw[g].values),
                "isotonic_transported": subgroup_cal(yw, np.clip(iso.predict(pw), 0, 1), Xw[g].values),
            }
        waves[str(yr)] = entry
        print("  %d: AUC=%.4f | raw ECE=%.4f -> iso ECE=%.4f | raw brier=%.4f -> iso brier=%.4f | "
              "max-group age ECE %.4f -> %.4f" % (
            yr, entry["roc_auc"], entry["raw"]["ece"], entry["isotonic_transported"]["ece"],
            entry["raw"]["brier"], entry["isotonic_transported"]["brier"],
            entry["subgroup_Age"]["raw"]["max_group_ece"],
            entry["subgroup_Age"]["isotonic_transported"]["max_group_ece"]), flush=True)
    out["per_wave_calibration"] = waves
    save_json(out, OUTPATH)

    # refit-per-wave comparison (what if calibrators were refit per wave?)
    print("\n=== Refit-per-wave comparison (upper bound on transport quality) ===", flush=True)
    refit = {}
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr)
        Xw = wy[FEATURES]; yw = wy[TARGET].astype(int)
        Xr, Xc, yr_, yc_ = train_test_split(Xw, yw, test_size=0.40, random_state=42, stratify=yw)
        mm = hgb(42); mm.fit(Xr, yr_)
        pc = mm.predict_proba(Xc)[:, 1]
        pt = mm.predict_proba(Xw)[:, 1]
        iso_w = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc, yc_.values)
        refit[str(yr)] = {"refit_isotonic": cal_block(yw.values, np.clip(iso_w.predict(pt), 0, 1)),
                          "transported_isotonic": cal_block(yw.values, np.clip(iso.predict(m.predict_proba(Xw)[:, 1]), 0, 1))}
        print("  %d: transported ECE=%.4f vs refit ECE=%.4f" % (
            yr, refit[str(yr)]["transported_isotonic"]["ece"], refit[str(yr)]["refit_isotonic"]["ece"]), flush=True)
    out["per_wave_refit_comparison"] = refit
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
