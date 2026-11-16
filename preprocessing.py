"""
Phase 1: Data Preprocessing for PaySim UPI fraud detection.

Loads PaySim dataset, cleans data, encodes features, applies SMOTE,
and saves train/test splits.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    DATA_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


def generate_sample_paysim(n_rows: int = 10000, output_path: Path | None = None) -> pd.DataFrame:
    """
    Generate synthetic PaySim-like data when the real dataset is unavailable.
    Useful for demos and local development.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_probs = [0.50, 0.20, 0.15, 0.10, 0.05]

    n_fraud = max(int(n_rows * 0.005), 50)
    fraud_idx = set(rng.choice(n_rows, size=n_fraud, replace=False))

    rows = []
    for i in range(n_rows):
        tx_type = rng.choice(types, p=type_probs)
        is_fraud = 1 if i in fraud_idx else 0
        amount = float(rng.exponential(5000) + (50000 if is_fraud else 0))
        old_org = float(rng.uniform(0, 500000))
        if is_fraud:
            new_org = old_org
        else:
            new_org = max(old_org - amount, 0)
        old_dest = float(rng.uniform(0, 500000))
        new_dest = old_dest + amount if not is_fraud else old_dest

        rows.append(
            {
                "step": rng.integers(1, 744),
                "type": tx_type,
                "amount": round(amount, 2),
                "nameOrig": f"C{i % 500:04d}",
                "oldbalanceOrg": round(old_org, 2),
                "newbalanceOrig": round(new_org, 2),
                "nameDest": f"C{(i + 137) % 500:04d}",
                "oldbalanceDest": round(old_dest, 2),
                "newbalanceDest": round(new_dest, 2),
                "isFraud": is_fraud,
                "isFlaggedFraud": 0,
            }
        )

    df = pd.DataFrame(rows)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load PaySim CSV or generate synthetic data if missing."""
    path = path or RAW_DATA_PATH
    if not path.exists():
        print(f"Dataset not found at {path}. Generating synthetic PaySim data...")
        return generate_sample_paysim(output_path=path)
    return pd.read_csv(path)


def preprocess(
    df: pd.DataFrame | None = None,
    apply_smote: bool = True,
    save: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns dict with X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names.
    """
    if df is None:
        df = load_dataset()

    df = df.copy()

    # Handle missing values
    numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "type" in df.columns:
        df["type"] = df["type"].fillna("PAYMENT")

    # Drop unnecessary columns
    drop_cols = [
        "step",
        "nameOrig",
        "nameDest",
        "isFlaggedFraud",
    ]
    account_cols = {"nameOrig": "sender", "nameDest": "receiver"}
    for orig, alias in account_cols.items():
        if orig in df.columns:
            df[alias] = df[orig]

    keep_accounts = df[["sender", "receiver"]].copy() if "sender" in df.columns else None

    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode transaction type
    le = LabelEncoder()
    df_model["type_encoded"] = le.fit_transform(df_model["type"].astype(str))

    # Normalize numerical features (base columns only at this stage)
    base_numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_encoded"]
    scaler = StandardScaler()
    df_model[base_numeric] = scaler.fit_transform(df_model[base_numeric])

    feature_cols = [c for c in base_numeric if c in df_model.columns]
    X = df_model[feature_cols].values
    y = df_model[TARGET_COLUMN].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if apply_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_cols,
        "raw_df": df,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "account_data": keep_accounts,
    }

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(result, f)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    return result


if __name__ == "__main__":
    preprocess()
