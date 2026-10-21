import os, sys, json, time, functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
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
OUTPATH = os.path.join(LOGS, "audit_results5.json")


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


def main():
    t0 = time.time()
    out = {}
    if os.path.exists(OUTPATH):
        out = json.load(open(OUTPATH))
    print("resuming with sections:", list(out.keys()), flush=True)

    df = load_year(2015)
    lab = df[TARGET].astype(int)
    Xrest, Xte, yrest, yte = train_test_split(df[FEATURES], lab, test_size=0.20, random_state=42, stratify=lab)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.20, random_state=42, stratify=yrest)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    Xcal = Xcal.reset_index(drop=True); ycal = ycal.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True); yv3 = yte.reset_index(drop=True).values

    m42 = hgb(42); m42.fit(Xtr, ytr)
    p_te = m42.predict_proba(Xte)[:, 1]

    # ============ E. SHAP STABILITY ============
    if "shap_stability" not in out:
        print("\n=== E. SHAP top-10 stability (Jaccard) ===", flush=True)
        import shap
        ranks = {}
        for seed in [42, 7, 123, 2024, 99]:
            mm = hgb(seed); mm.fit(Xtr, ytr)
            sv = shap.TreeExplainer(mm).shap_values(Xte.sample(2000, random_state=42))
            if isinstance(sv, list): sv = sv[1]
            order = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
            ranks["seed_%d" % seed] = order[:10]
            print("  seed %d top10: %s" % (seed, ", ".join(order[:10])), flush=True)
        for yr in [2017, 2019, 2021]:
            wy = load_year(yr)
            sv = shap.TreeExplainer(m42).shap_values(wy[FEATURES].sample(2000, random_state=42))
            if isinstance(sv, list): sv = sv[1]
            order = [FEATURES[i] for i in np.argsort(np.abs(np.array(sv)).mean(axis=0))[::-1]]
            ranks["wave_%d" % yr] = order[:10]
            print("  wave %d top10: %s" % (yr, ", ".join(order[:10])), flush=True)
        def jac(a, b):
            sa, sb = set(a), set(b)
            return len(sa & sb) / max(len(sa | sb), 1)
        seeds_only = [k for k in ranks if k.startswith("seed_")]
        wave_only = [k for k in ranks if k.startswith("wave_")]
        j_seed = [jac(ranks["seed_42"], ranks[k]) for k in seeds_only]
        j_wave = [jac(ranks["seed_42"], ranks[k]) for k in wave_only]
        out["shap_stability"] = {
            "rankings": ranks,
            "mean_jaccard_across_seeds": float(np.mean(j_seed)),
            "min_jaccard_across_seeds": float(np.min(j_seed)),
            "mean_jaccard_2015_vs_waves": float(np.mean(j_wave)),
            "min_jaccard_2015_vs_waves": float(np.min(j_wave)),
        }
        print("  Jaccard across seeds: mean=%.3f min=%.3f" % (
            out["shap_stability"]["mean_jaccard_across_seeds"],
            out["shap_stability"]["min_jaccard_across_seeds"]), flush=True)
        print("  Jaccard 2015 vs waves: mean=%.3f min=%.3f" % (
            out["shap_stability"]["mean_jaccard_2015_vs_waves"],
            out["shap_stability"]["min_jaccard_2015_vs_waves"]), flush=True)
        save_json(out, OUTPATH)

    # ============ F. CONSTRAINED THRESHOLD OPTIMIZATION ============
    if "constrained_threshold" not in out:
        print("\n=== F. Constrained threshold optimization (min FN s.t. bounded age TPR gap) ===", flush=True)
        ages = Xte["Age"].values
        grid = []
        for t in np.linspace(0.05, 0.95, 91):
            pred = (p_te >= t).astype(int)
            fn = int(((pred == 0) & (yv3 == 1)).sum())
            per = {}
            for a in sorted(set(ages)):
                msk = ages == a
                npos = int((yv3[msk] == 1).sum())
                if npos < 20: continue
                per[a] = float(((pred[msk] == 1) & (yv3[msk] == 1)).sum() / npos)
            gap = (max(per.values()) - min(per.values())) if per else float("nan")
            grid.append({"t": float(t), "fn": fn, "age_tpr_gap": float(gap)})
        cons = {}
        for delta in [0.05, 0.10, 0.20, 0.30, 0.50, 1.01]:
            feas = [g for g in grid if g["age_tpr_gap"] <= delta]
            if feas:
                best = min(feas, key=lambda g: g["fn"])
                cons["%.2f" % delta] = {"feasible": True, "threshold": best["t"], "fn": best["fn"],
                                        "achieved_gap": best["age_tpr_gap"],
                                        "fn_rate": best["fn"] / max(int((yv3 == 1).sum()), 1)}
            else:
                cons["%.2f" % delta] = {"feasible": False}
            print("  max age TPR gap <= %.2f : %s" % (delta, cons["%.2f" % delta]), flush=True)
        out["constrained_threshold"] = {"grid": grid, "by_delta": cons}
        save_json(out, OUTPATH)

    # ============ G. DESIGN-EFFECT SENSITIVITY ============
    if "design_effect_sensitivity" not in out:
        print("\n=== G. Design-effect sensitivity for subgroup CIs ===", flush=True)
        prev = json.load(open(os.path.join(LOGS, "audit_results.json")))
        fair = prev.get("fairness_diagnostics", {})
        deff = {}
        for d in [1.0, 1.5, 2.0, 3.0]:
            mult = float(np.sqrt(d))
            entry = {"multiplier": mult, "groups": {}}
            for g, v in fair.items():
                w = v["bootstrap"]["tpr_gap_ci95"][1] - v["bootstrap"]["tpr_gap_ci95"][0]
                entry["groups"][g] = {"tpr_gap_ci_width": float(w * mult),
                                      "tpr_gap_ci95_design_scaled": [v["bootstrap"]["tpr_gap_ci95"][0] * mult,
                                                                    v["bootstrap"]["tpr_gap_ci95"][1] * mult]}
            worst = max(entry["groups"].items(), key=lambda kv: kv[1]["tpr_gap_ci_width"])
            entry["widest_group"] = {"stratifier": worst[0], "ci_width": worst[1]["tpr_gap_ci_width"]}
            deff[str(d)] = entry
            print("  DEFF=%.1f -> widest TPR-gap CI width = %.4f (%s)" % (
                d, worst[1]["tpr_gap_ci_width"], worst[0]), flush=True)
        out["design_effect_sensitivity"] = {
            "assumption": "CI widths scale by sqrt(DEFF) under clustered/stratified survey sampling; "
                          "the cleaned release carries no design variables, so DEFF is assumed, not estimated",
            "by_deff": deff,
        }
        save_json(out, OUTPATH)

    # ============ H. SEVERITY-STRATIFIED SENSITIVITY ============
    if "severity_sensitivity" not in out:
        print("\n=== H. Severity-stratified sensitivity (positive cases) ===", flush=True)
        sev = {}
        gh = Xte["GenHlth"].values
        bmi = Xte["BMI"].values
        pos = yv3 == 1
        strata = {
            "GenHlth 1-2 (good/very good)": pos & (gh <= 2),
            "GenHlth 3 (fair)": pos & (gh == 3),
            "GenHlth 4-5 (poor/very poor)": pos & (gh >= 4),
            "BMI < 30": pos & (bmi < 30),
            "BMI >= 30": pos & (bmi >= 30),
        }
        for name, msk in strata.items():
            if msk.sum() < 100: continue
            y_s = yv3[msk]; p_s = p_te[msk]
            try:
                auc = roc_auc_score(y_s, p_s)
            except Exception:
                auc = float("nan")
            sev[name] = {"n": int(msk.sum()), "within_stratum_auc": float(auc),
                         "mean_pred": float(p_s.mean())}
            print("  %-30s n=%6d within-stratum AUC=%.4f mean_pred=%.3f" % (
                name, sev[name]["n"], auc, sev[name]["mean_pred"]), flush=True)
        out["severity_sensitivity"] = sev
        save_json(out, OUTPATH)

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, OUTPATH)
    print("\nTotal elapsed: %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
