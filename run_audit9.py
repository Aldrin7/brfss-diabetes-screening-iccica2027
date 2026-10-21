import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import ensure_dir, save_json
from preprocessing import FEATURES, TARGET

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
MULTI = os.path.join(RAW, "brfss_multi")
LOGS = os.path.join(ROOT, "results", "logs")
ensure_dir(LOGS)
OUTPATH = os.path.join(LOGS, "audit_results9.json")


def load_year(year):
    if year == 2015:
        return pd.read_csv(os.path.join(RAW, "diabetes_binary_health_indicators_BRFSS2015.csv"))
    return pd.read_csv(os.path.join(MULTI, "cdc_brfss_diabetes_%d.csv" % year))


def hgb(seed=42):
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=seed)


def predict_chunked(model, X, chunk=40000):
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = model.predict_proba(X.iloc[i:i + chunk])[:, 1]
    return out


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
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)), "ece": e, "mce": mc,
            "mean_pred": float(p.mean()), "observed": float(y.mean())}


def main():
    t0 = time.time()
    out = {}
    df = load_year(2015)
    lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    m = hgb(42); m.fit(Xtr, ytr)
    p_cal = m.predict_proba(Xcal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, ycal.values)

    print("=== Transport vs refit calibration per wave (chunked prediction) ===", flush=True)
    res = {}
    for yr in [2017, 2019, 2021]:
        wy = load_year(yr)
        Xw = wy[FEATURES].reset_index(drop=True)
        yw = wy[TARGET].astype(int).values
        p_tr = predict_chunked(m, Xw)
        transported = cal_block(yw, np.clip(iso.predict(p_tr), 0, 1))
        # refit: train on 60% of the wave, calibrate on 20%, evaluate on 20%
        Xa, Xb, ya, yb = train_test_split(Xw, yw, test_size=0.40, random_state=42, stratify=yw)
        mm = hgb(42); mm.fit(Xa, ya)
        pc = predict_chunked(mm, Xb)
        iso_w = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc, yb)
        pt = predict_chunked(mm, Xw)
        refit = cal_block(yw, np.clip(iso_w.predict(pt), 0, 1))
        res[str(yr)] = {"transported_isotonic": transported, "refit_isotonic": refit,
                        "transport_penalty_ece": transported["ece"] - refit["ece"],
                        "transport_penalty_brier": transported["brier"] - refit["brier"]}
        print("  %d: transported ECE=%.4f brier=%.4f | refit ECE=%.4f brier=%.4f | penalty ECE %+.4f" % (
            yr, transported["ece"], transported["brier"], refit["ece"], refit["brier"],
            res[str(yr)]["transport_penalty_ece"]), flush=True)
    out["per_wave_transport_vs_refit"] = res
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
