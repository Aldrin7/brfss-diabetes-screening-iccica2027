import os, sys, json, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import set_seed, ensure_dir
from preprocessing import load_raw, split_data, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "diabetes_binary_health_indicators_BRFSS2015.csv")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
ensure_dir(FIGS)

# confusion matrix on seed 42
set_seed(42)
df = load_raw(DATA)
X_tr, X_te, y_tr, y_te = split_data(df, seed=42, test_size=0.2)
m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced", random_state=42)
m.fit(X_tr, y_tr)
pred = m.predict(X_te)
cm = confusion_matrix(y_te, pred, labels=[0,1])
fig, ax = plt.subplots(figsize=(5,4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No diabetes","Diabetes/prediabetes"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion matrix (HGB, seed 42, test set)")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "confusion_matrix.png"), dpi=300)
plt.close(fig)
print("cm", cm.tolist())

# fairness aggregate save
d = json.load(open(os.path.join(RESULTS, "logs", "results.json")))
import statistics
fair = {}
for grp in ["Sex","Age","Income","Education"]:
    tprs=[r["models"]["HistGradientBoosting"]["fairness_"+grp+"_gaps"]["tpr_gap"] for r in d["runs"]]
    accs=[r["models"]["HistGradientBoosting"]["fairness_"+grp+"_gaps"]["accuracy_gap"] for r in d["runs"]]
    fair[grp] = {
        "tpr_gap_mean": statistics.mean(tprs), "tpr_gap_std": statistics.stdev(tprs),
        "acc_gap_mean": statistics.mean(accs), "acc_gap_std": statistics.stdev(accs),
    }
json.dump(fair, open(os.path.join(RESULTS,"logs","fairness_agg.json"),"w"), indent=2)
print(json.dumps(fair, indent=2))