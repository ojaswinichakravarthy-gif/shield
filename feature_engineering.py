"""
Phase 2: Feature Engineering for UPI fraud detection.

Generates domain-specific features and merges them with preprocessed data.
"""

import pickle

import numpy as np
import pandas as pd

from config import FEATURES_PATH, PROCESSED_DATA_PATH


def engineer_features(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """
    Generate engineered features from raw transaction data.

    Features:
        - balance_difference_sender
        - balance_difference_receiver
        - transaction_frequency (per sender)
        - average_transaction_amount (per sender)
        - sender_out_degree
        - receiver_in_degree
    """
    if df is None:
        with open(PROCESSED_DATA_PATH, "rb") as f:
            processed = pickle.load(f)
        df = processed["raw_df"].copy()
    else:
        df = df.copy()

    # Balance differences
    df["balance_difference_sender"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_difference_receiver"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Sender-level aggregates
    sender_stats = (
        df.groupby("sender")
        .agg(
            transaction_frequency=("amount", "count"),
            average_transaction_amount=("amount", "mean"),
        )
        .reset_index()
    )
    df = df.merge(sender_stats, on="sender", how="left")

    # Graph degree features
    out_degree = df.groupby("sender").size().rename("sender_out_degree")
    in_degree = df.groupby("receiver").size().rename("receiver_in_degree")

    df = df.join(out_degree, on="sender")
    df = df.join(in_degree, on="receiver")

    df["transaction_frequency"] = df["transaction_frequency"].fillna(1)
    df["average_transaction_amount"] = df["average_transaction_amount"].fillna(df["amount"].median())
    df["sender_out_degree"] = df["sender_out_degree"].fillna(1)
    df["receiver_in_degree"] = df["receiver_in_degree"].fillna(1)

    engineered_cols = [
        "balance_difference_sender",
        "balance_difference_receiver",
        "transaction_frequency",
        "average_transaction_amount",
        "sender_out_degree",
        "receiver_in_degree",
    ]

    if save:
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURES_PATH, "wb") as f:
            pickle.dump({"df": df, "engineered_cols": engineered_cols}, f)
        print(f"Engineered features saved to {FEATURES_PATH}")

    return df


def build_full_feature_matrix(processed: dict, engineered_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine preprocessed base features with engineered features for modeling."""
    from sklearn.preprocessing import StandardScaler

    from config import FEATURE_COLUMNS, TARGET_COLUMN

    df = engineered_df.copy()
    eng_cols = [
        "balance_difference_sender",
        "balance_difference_receiver",
        "transaction_frequency",
        "average_transaction_amount",
        "sender_out_degree",
        "receiver_in_degree",
    ]

    scaler_eng = StandardScaler()
    df[eng_cols] = scaler_eng.fit_transform(df[eng_cols])

    # Re-encode type using the same encoder from preprocessing
    le = processed["label_encoder"]
    df["type_encoded"] = le.transform(df["type"].astype(str))

    # Normalize base numeric columns using the fitted preprocessing scaler
    base_numeric = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "type_encoded",
    ]
    scaler = processed["scaler"]
    df[base_numeric] = scaler.transform(df[base_numeric])

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values.astype(int)

    return X, y, feature_cols


if __name__ == "__main__":
    engineer_features()
