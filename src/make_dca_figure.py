import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
FIGS = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

d = {}
for f in sorted(glob.glob(os.path.join(LOGS, "audit_results*.json"))):
    try:
        d.update(json.load(open(f)))
    except Exception:
        pass

dc = d.get("decision_curve")
if dc:
    t = np.array(dc["thresholds"])
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    ax.plot(t, dc["model"], "-o", ms=2.5, color="steelblue", label="HGB model")
    ax.plot(t, dc["treat_all"], "--s", ms=2.5, color="firebrick", label="Treat all")
    ax.plot(t, dc["treat_none"], ":", color="gray", lw=1.5, label="Treat none")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.15, 0.12)
    ax.set_title("Decision-curve net benefit (HGB, 2015 test)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "decision_curve.png"), dpi=300)
    plt.close(fig)
    print("saved decision_curve.png")
else:
    print("no decision_curve in results")
