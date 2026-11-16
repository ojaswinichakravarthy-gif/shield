"""
UPI Generalization Study

Train:
    PaySim Dataset

Test:
    External UPI Dataset

Purpose:
    Measure how well SHIELD-AI generalizes to unseen UPI transaction data.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from ensemble import get_predictor
from config import PROCESSED_DATA_PATH


# ==========================================================
# LOAD EXTERNAL UPI DATASET
# ==========================================================

UPI_DATASET = "data/upi_transactions_2024.csv"

df = pd.read_csv(UPI_DATASET)

print("=" * 60)
print("UPI GENERALIZATION STUDY")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFraud Distribution:")
print(df["fraud_flag"].value_counts())


# ==========================================================
# LOAD PAYSIM SCALER
# ==========================================================

with open(PROCESSED_DATA_PATH, "rb") as f:
    processed = pickle.load(f)

scaler = processed["scaler"]


# ==========================================================
# FEATURE ADAPTER
# ==========================================================

print("\nBuilding compatible SHIELD features...")

amount = df["amount (INR)"]

hour = df["hour_of_day"]

weekend = df["is_weekend"]

device_risk = (
    df["device_type"]
    .astype(str)
    .str.lower()
    .map({
        "android": 0,
        "ios": 0,
        "web": 1
    })
    .fillna(0)
)

network_risk = (
    df["network_type"]
    .astype(str)
    .str.lower()
    .map({
        "4g": 0,
        "5g": 0,
        "wifi": 1
    })
    .fillna(0)
)

merchant_risk = (
    pd.factorize(df["merchant_category"])[0]
)

transaction_type = (
    pd.factorize(df["transaction type"])[0]
)

sender_bank = (
    pd.factorize(df["sender_bank"])[0]
)

receiver_bank = (
    pd.factorize(df["receiver_bank"])[0]
)


# ==========================================================
# BUILD PAYSIM-SHAPED FEATURE MATRIX
# ==========================================================

X = pd.DataFrame({
    "amount": amount,

    "oldbalanceOrg": amount * 5,
    "newbalanceOrig": amount * 4,

    "oldbalanceDest": amount * 3,
    "newbalanceDest": amount * 4,

    "type_encoded": transaction_type,

    "balance_difference_sender": amount,

    "balance_difference_receiver": amount,

    "transaction_frequency": hour + 1,

    "average_transaction_amount": amount,

    "sender_out_degree": sender_bank,

    "receiver_in_degree": receiver_bank,
})


# ==========================================================
# SCALE FIRST 6 FEATURES
# ==========================================================

base_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_encoded",
]

scaled = scaler.transform(X[base_cols])

X_scaled = X.copy()

for i, col in enumerate(base_cols):
    X_scaled[col] = scaled[:, i]

X_final = X_scaled.values


# ==========================================================
# CREATE DUMMY ACCOUNTS
# ==========================================================

senders = [
    f"SENDER_{i}"
    for i in range(len(df))
]

receivers = [
    f"RECEIVER_{i}"
    for i in range(len(df))
]


# ==========================================================
# RUN SHIELD-AI
# ==========================================================

print("\nRunning SHIELD-AI...")

predictor = get_predictor()

results = predictor.predict_batch(
    X_final,
    senders,
    receivers
)

scores = results["ensemble_scores"]

preds = (scores > 0.5).astype(int)

y_true = df["fraud_flag"].values


# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds, zero_division=0)
recall = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)

try:
    auc = roc_auc_score(y_true, scores)
except:
    auc = 0


print("\n")
print("=" * 60)
print("GENERALIZATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")


# ==========================================================
# INTERPRETATION
# ==========================================================

print("\n")
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if f1 > 0.80:
    print("Excellent generalization.")
elif f1 > 0.60:
    print("Good generalization.")
elif f1 > 0.40:
    print("Moderate generalization.")
else:
    print("Weak generalization. Domain shift detected.")