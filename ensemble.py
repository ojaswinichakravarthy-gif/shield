"""
Phase 6: Ensemble Fusion — weighted combination of tabular + GNN predictions.
"""

import pickle
from typing import Any

import numpy as np
import torch

from config import ENSEMBLE_WEIGHTS, FRAUD_THRESHOLD, GRAPH_DATA_PATH, GRAPH_META_PATH
from models.gnn_model import load_gnn, transaction_gnn_score
from models.tabular_models import TabularModelSuite


def risk_level(score: float) -> str:
    """Map fraud probability to human-readable risk level."""
    if score >= 0.8:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


class EnsemblePredictor:
    """
    Combines Logistic Regression, Random Forest, XGBoost, and GNN scores.

    Final Score = 0.15*LR + 0.25*RF + 0.30*XGB + 0.30*GNN
    """

    def __init__(self):
        self.tabular = TabularModelSuite.load()
        self.gnn_model, _ = load_gnn()
        self.graph_data = torch.load(GRAPH_DATA_PATH, weights_only=False)
        with open(GRAPH_META_PATH, "rb") as f:
            self.graph_meta = pickle.load(f)

    def predict_batch(self, X: np.ndarray, senders: list[str], receivers: list[str]) -> dict[str, Any]:
        """Batch prediction with per-model and ensemble scores."""
        tabular_probs = self.tabular.predict_proba(X)
        n = len(X)
        gnn_probs = np.zeros(n)

        for i in range(n):
            gnn_probs[i] = transaction_gnn_score(
                self.gnn_model, self.graph_data, self.graph_meta, senders[i], receivers[i]
            )

        ensemble_scores = (
            ENSEMBLE_WEIGHTS["logistic_regression"] * tabular_probs["logistic_regression"]
            + ENSEMBLE_WEIGHTS["random_forest"] * tabular_probs["random_forest"]
            + ENSEMBLE_WEIGHTS["xgboost"] * tabular_probs["xgboost"]
            + ENSEMBLE_WEIGHTS["gnn"] * gnn_probs
        )

        predictions = np.where(ensemble_scores > FRAUD_THRESHOLD, "Fraud", "Legitimate")

        return {
            "ensemble_scores": ensemble_scores,
            "predictions": predictions,
            "lr_probs": tabular_probs["logistic_regression"],
            "rf_probs": tabular_probs["random_forest"],
            "xgb_probs": tabular_probs["xgboost"],
            "gnn_probs": gnn_probs,
        }

    def predict_single(
        self,
        features: np.ndarray,
        sender: str,
        receiver: str,
    ) -> dict[str, Any]:
        """Single transaction prediction."""
        result = self.predict_batch(
            features.reshape(1, -1),
            [sender],
            [receiver],
        )
        score = float(result["ensemble_scores"][0])
        return {
            "fraud_probability": round(score, 4),
            "prediction": result["predictions"][0],
            "risk_level": risk_level(score),
            "model_scores": {
                "logistic_regression": float(result["lr_probs"][0]),
                "random_forest": float(result["rf_probs"][0]),
                "xgboost": float(result["xgb_probs"][0]),
                "gnn": float(result["gnn_probs"][0]),
            },
        }


_predictor: EnsemblePredictor | None = None


def get_predictor() -> EnsemblePredictor:
    """Lazy-load ensemble predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor
