import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_models(seed=42):
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=20,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
        class_weight="balanced", random_state=seed,
    )
    return {"LogisticRegression": lr, "RandomForest": rf, "HistGradientBoosting": hgb}
