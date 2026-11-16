import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


print("=" * 60)
print("Credit Card Fraud Detection - Baseline Models")
print("=" * 60)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("research_archive/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"\nDataset Shape: {df.shape}")
print("\nClass Distribution:")
print(y.value_counts())

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# SMOTE (TRAIN ONLY)
# =====================================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Training Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =====================================================
# MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ),
}

results = {}

# =====================================================
# TRAIN + EVALUATE
# =====================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    model.fit(X_train_smote, y_train_smote)

    preds = model.predict(X_test_scaled)

    probs = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results).T

print(results_df.round(4))

best_model = results_df["F1"].idxmax()

print("\nBest Model (by F1 Score):", best_model)

results_df.to_csv("research_archive/creditcard_model_results.csv")

print("\nResults saved to:")
print("research_archive/creditcard_model_results.csv")