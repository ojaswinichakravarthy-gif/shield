import os
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

from config import FEATURES_PATH, PROCESSED_DATA_PATH
from ensemble import EnsemblePredictor
from feature_engineering import build_full_feature_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

with open(PROCESSED_DATA_PATH, "rb") as f:
    processed = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    eng = pickle.load(f)

X, y, _ = build_full_feature_matrix(processed, eng["df"])

test_idx = processed["test_idx"]

X_test = X[test_idx]
y_test = y[test_idx]

df_meta = eng["df"]

senders = df_meta.iloc[test_idx]["sender"].astype(str).tolist()
receivers = df_meta.iloc[test_idx]["receiver"].astype(str).tolist()

predictor = EnsemblePredictor()

prediction_results = predictor.predict_batch(
    X_test,
    senders,
    receivers
)

lr_probs = prediction_results["lr_probs"]
rf_probs = prediction_results["rf_probs"]
xgb_probs = prediction_results["xgb_probs"]
gnn_probs = prediction_results["gnn_probs"]

results = {
    "lr_probs": lr_probs,
    "rf_probs": rf_probs,
    "xgb_probs": xgb_probs,
    "gnn_probs": gnn_probs,
}

ablations = {
    "Full Ensemble":
        0.15 * results["lr_probs"] +
        0.25 * results["rf_probs"] +
        0.30 * results["xgb_probs"] +
        0.30 * results["gnn_probs"],

    "Without LR":
        (0.25 * results["rf_probs"] +
         0.30 * results["xgb_probs"] +
         0.30 * results["gnn_probs"]) / 0.85,

    "Without RF":
        (0.15 * results["lr_probs"] +
         0.30 * results["xgb_probs"] +
         0.30 * results["gnn_probs"]) / 0.75,

    "Without XGB":
        (0.15 * results["lr_probs"] +
         0.25 * results["rf_probs"] +
         0.30 * results["gnn_probs"]) / 0.70,

    "Without GNN":
        (0.15 * results["lr_probs"] +
         0.25 * results["rf_probs"] +
         0.30 * results["xgb_probs"]) / 0.70,
}
ablation_results = []

for name, scores in ablations.items():

    preds = (scores > 0.5).astype(int)

    ablation_results.append({
        "Configuration": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC_AUC": roc_auc_score(y_test, scores)
    })
  

df = pd.DataFrame(ablation_results)

print("\n")
print("=" * 80)
print("ABLATION STUDY")
print("=" * 80)
print(df)

df.to_csv("ablation_results.csv", index=False)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(
    df["Configuration"],
    df["F1"]
)

plt.ylabel("F1 Score")
plt.title("SHIELD-AI Ablation Study")

for i,v in enumerate(df["F1"]):
    plt.text(i, v, f"{v:.4f}", ha="center")

plt.tight_layout()
plt.savefig("ablation_f1.png", dpi=300)