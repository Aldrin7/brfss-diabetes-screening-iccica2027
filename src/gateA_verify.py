import pandas as pd
import json
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = root + r"\data\raw\diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(p)

report = {}
report["shape"] = list(df.shape)
report["columns"] = list(df.columns)
report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
report["missing_total"] = int(df.isna().sum().sum())
report["missing_per_col"] = {c: int(v) for c, v in df.isna().sum().items()}
report["target_counts"] = {str(k): int(v) for k, v in df["Diabetes_binary"].value_counts().items()}
report["prevalence"] = float(df["Diabetes_binary"].mean())
report["sex_counts"] = {str(k): int(v) for k, v in df["Sex"].value_counts().items()}
report["age_counts"] = {str(k): int(v) for k, v in df["Age"].value_counts().sort_index().items()}
report["income_counts"] = {str(k): int(v) for k, v in df["Income"].value_counts().sort_index().items()}
report["edu_counts"] = {str(k): int(v) for k, v in df["Education"].value_counts().sort_index().items()}
report["duplicates"] = int(df.duplicated().sum())
report["n_unique_rows"] = int(df.drop_duplicates().shape[0])

print("shape:", report["shape"])
print("missing_total:", report["missing_total"])
print("target_counts:", report["target_counts"])
print("prevalence:", round(report["prevalence"], 4))
print("duplicates:", report["duplicates"])
print("sex:", report["sex_counts"])
print("age:", report["age_counts"])
print("income:", report["income_counts"])
print("edu:", report["edu_counts"])

os.makedirs(root + r"\results\logs", exist_ok=True)
with open(root + r"\results\logs\gateA_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("saved gateA_report.json")
