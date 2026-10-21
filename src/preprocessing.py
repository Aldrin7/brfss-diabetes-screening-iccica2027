import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET = "Diabetes_binary"
SENSITIVE = ["Sex", "Age", "Income", "Education"]

CONTINUOUS = ["BMI", "MentHlth", "PhysHlth", "GenHlth"]

def load_raw(path):
    df = pd.read_csv(path)
    assert "Diabetes_binary" in df.columns, "target missing"
    assert "ID" in df.columns or "Id" in df.columns or True
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    for c in FEATURES + [TARGET]:
        assert c in df.columns, f"missing col {c}"
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0, "missing values present"
    return df

def split_data(df, seed=42, test_size=0.2):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr.reset_index(drop=True), X_te.reset_index(drop=True), y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

def make_cv(X, y, n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
