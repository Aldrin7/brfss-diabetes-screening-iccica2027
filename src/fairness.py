import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    m["tn"]=int(tn); m["fp"]=int(fp); m["fn"]=int(fn); m["tp"]=int(tp)
    return m

def group_metrics(y_true, y_pred, groups):
    out = {}
    for g in sorted(set(groups)):
        mask = (groups == g)
        if mask.sum() < 20:
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        out[str(g)] = {
            "n": int(mask.sum()),
            "prevalence": float(yt.mean()),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "tpr": float(recall_score(yt, yp, zero_division=0)),
            "fpr": float(((yp==1)&(yt==0)).sum()/max((yt==0).sum(),1)),
        }
    return out

def fairness_gaps(group_m):
    # demographic parity diff = max-min positive prediction rate (approx via recall? use selection rate)
    # We'll compute selection rates separately; here compute TPR gap and accuracy gap
    tprs = [v["tpr"] for v in group_m.values()]
    accs = [v["accuracy"] for v in group_m.values()]
    return {
        "tpr_gap": float(max(tprs)-min(tprs)) if tprs else float("nan"),
        "accuracy_gap": float(max(accs)-min(accs)) if accs else float("nan"),
        "min_group_n": int(min(v["n"] for v in group_m.values())) if group_m else 0,
    }
