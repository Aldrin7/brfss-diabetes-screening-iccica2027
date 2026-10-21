import os, sys, json, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed, ensure_dir, save_json
from preprocessing import load_raw, split_data, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
ensure_dir(FIGS)

def main():
    set_seed(42)
    df = load_raw(DATA)
    X_tr, X_te, y_tr, y_te = split_data(df, seed=42, test_size=0.2)
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    # SHAP on a sample of test set
    sample_n = 2000
    Xs = X_te.sample(n=sample_n, random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    # sv may be list for binary; take class 1
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [(FEATURES[i], float(mean_abs[i])) for i in order]
    print("Top features by mean |SHAP|:")
    for f, v in top[:15]:
        print(f"  {f}: {v:.5f}")
    save_json({"top_features": top, "sample_n": sample_n}, os.path.join(RESULTS, "logs", "shap.json"))
    # figure: bar of top 12
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [t[0] for t in top[:12]][::-1]
    vals = [t[1] for t in top[:12]][::-1]
    ax.barh(names, vals, color="steelblue")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Top 12 features (HistGradientBoosting, test sample n=2000)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "shap_top_features.png"), dpi=300)
    plt.close(fig)
    print("saved shap figure")
    # beeswarm-ish: save a summary plot
    try:
        shap.summary_plot(sv, Xs, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("saved shap summary")
    except Exception as e:
        print("summary plot failed:", e)

if __name__ == "__main__":
    main()
