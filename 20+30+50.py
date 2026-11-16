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
print("Credit Card Fraud Detection - Ensemble Experiment")
print("=" * 60)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("research_archive/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALE
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# SMOTE
# =====================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nTraining Distribution After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# =====================================================
# TRAIN MODELS
# =====================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

print("\nTraining Logistic Regression...")
lr.fit(X_train_smote, y_train_smote)

print("Training Random Forest...")
rf.fit(X_train_smote, y_train_smote)

print("Training XGBoost...")
xgb.fit(X_train_smote, y_train_smote)

# =====================================================
# PROBABILITIES
# =====================================================

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# WEIGHTED ENSEMBLE
# =====================================================

ensemble_probs = (
    0.20 * lr_probs +
    0.30 * rf_probs +
    0.50 * xgb_probs
)

ensemble_preds = (ensemble_probs > 0.5).astype(int)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, ensemble_preds)
precision = precision_score(y_test, ensemble_preds)
recall = recall_score(y_test, ensemble_preds)
f1 = f1_score(y_test, ensemble_preds)
roc_auc = roc_auc_score(y_test, ensemble_probs)

print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# =====================================================
# COMPARISON TABLE
# =====================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Ensemble"
    ],
    "Precision": [
        precision_score(y_test, (lr_probs > 0.5).astype(int)),
        precision_score(y_test, (rf_probs > 0.5).astype(int)),
        precision_score(y_test, (xgb_probs > 0.5).astype(int)),
        precision
    ],
    "Recall": [
        recall_score(y_test, (lr_probs > 0.5).astype(int)),
        recall_score(y_test, (rf_probs > 0.5).astype(int)),
        recall_score(y_test, (xgb_probs > 0.5).astype(int)),
        recall
    ],
    "F1": [
        f1_score(y_test, (lr_probs > 0.5).astype(int)),
        f1_score(y_test, (rf_probs > 0.5).astype(int)),
        f1_score(y_test, (xgb_probs > 0.5).astype(int)),
        f1
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, lr_probs),
        roc_auc_score(y_test, rf_probs),
        roc_auc_score(y_test, xgb_probs),
        roc_auc
    ]
})

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.round(4))

comparison.to_csv(
    "research_archive/creditcard_ensemble_results2.csv",
    index=False
)

print("\nResults saved to research_archive/creditcard_ensemble_results2.csv")

pred_df = pd.DataFrame({
    "y_true": y_test,
    "ensemble_prob": ensemble_probs
})

pred_df.to_csv(
    "research_archive/creditcard_ensemble_predictions.csv",
    index=False
)