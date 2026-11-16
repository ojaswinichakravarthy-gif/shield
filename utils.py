"""Utility functions for feature vector construction from raw transaction input."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import FEATURE_COLUMNS, FEATURES_PATH, PROCESSED_DATA_PATH


def build_feature_vector(
    sender: str,
    receiver: str,
    amount: float,
    tx_type: str,
    oldbalance_org: float | None = None,
    newbalance_orig: float | None = None,
    oldbalance_dest: float | None = None,
    newbalance_dest: float | None = None,
) -> np.ndarray:
    """
    Build a feature vector for a single transaction using stored scalers and stats.
    """
    with open(PROCESSED_DATA_PATH, "rb") as f:
        processed = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        eng = pickle.load(f)

    df = eng["df"]
    scaler: StandardScaler = processed["scaler"]
    le: LabelEncoder = processed["label_encoder"]

    oldbalance_org = oldbalance_org if oldbalance_org is not None else float(df["oldbalanceOrg"].median())
    newbalance_orig = newbalance_orig if newbalance_orig is not None else max(oldbalance_org - amount, 0)
    oldbalance_dest = oldbalance_dest if oldbalance_dest is not None else float(df["oldbalanceDest"].median())
    newbalance_dest = newbalance_dest if newbalance_dest is not None else oldbalance_dest + amount

    balance_diff_sender = oldbalance_org - newbalance_orig
    balance_diff_receiver = newbalance_dest - oldbalance_dest

    sender_rows = df[df["sender"] == sender]
    if len(sender_rows) > 0:
        tx_freq = len(sender_rows) + 1
        avg_amount = sender_rows["amount"].mean()
        out_degree = sender_rows["sender_out_degree"].iloc[0] + 1
    else:
        tx_freq = 1
        avg_amount = amount
        out_degree = 1

    receiver_rows = df[df["receiver"] == receiver]
    in_degree = receiver_rows["receiver_in_degree"].iloc[0] + 1 if len(receiver_rows) > 0 else 1

    if tx_type in le.classes_:
        type_encoded = int(le.transform([tx_type])[0])
    else:
        type_encoded = 0

    raw = {
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "type_encoded": type_encoded,
        "balance_difference_sender": balance_diff_sender,
        "balance_difference_receiver": balance_diff_receiver,
        "transaction_frequency": tx_freq,
        "average_transaction_amount": avg_amount,
        "sender_out_degree": out_degree,
        "receiver_in_degree": in_degree,
    }

    base_numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_encoded"]
    eng_numeric = [
        "balance_difference_sender",
        "balance_difference_receiver",
        "transaction_frequency",
        "average_transaction_amount",
        "sender_out_degree",
        "receiver_in_degree",
    ]

    base_df = pd.DataFrame(
    [[raw[c] for c in base_numeric]],
    columns=base_numeric
)

    base_vals = scaler.transform(base_df)[0]
    eng_vals = np.array([raw[c] for c in eng_numeric], dtype=float)

    # Normalize engineered features using dataset stats
    eng_df_vals = df[eng_numeric].values
    eng_mean = eng_df_vals.mean(axis=0)
    eng_std = eng_df_vals.std(axis=0) + 1e-6
    eng_vals = (eng_vals - eng_mean) / eng_std

    feature_dict = dict(zip(base_numeric, base_vals))
    for c, v in zip(eng_numeric, eng_vals):
        feature_dict[c] = v

    return np.array([feature_dict[c] for c in FEATURE_COLUMNS if c in feature_dict], dtype=float)


def load_metrics() -> dict[str, Any]:
    """Load evaluation metrics from disk."""
    from config import METRICS_PATH

    if not METRICS_PATH.exists():
        return {}
    import json

    with open(METRICS_PATH) as f:
        return json.load(f)
