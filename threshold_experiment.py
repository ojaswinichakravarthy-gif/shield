import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# Load your saved ensemble probabilities
df = pd.read_csv("research_archive/creditcard_ensemble_predictions.csv")

# Must contain:
# y_true
# ensemble_prob

y_true = df["y_true"]
probs = df["ensemble_prob"]

results = []

for threshold in np.arange(0.05, 1.00, 0.01):

    preds = (probs >= threshold).astype(int)

    results.append({
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    })

results_df = pd.DataFrame(results)

best_row = results_df.loc[results_df["f1"].idxmax()]

print("=" * 60)
print("BEST THRESHOLD")
print("=" * 60)
print(best_row)

print("\nTop 10 Thresholds By F1 Score")
print(
    results_df.sort_values(
        by="f1",
        ascending=False
    ).head(10)
)

results_df.to_csv(
    "research/threshold_results.csv",
    index=False
)

print("\nSaved:")
print("research/threshold_results.csv")