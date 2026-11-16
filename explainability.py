"""
Phase 7: Explainability — SHAP for XGBoost, graph-based explanations for GNN.
"""

import pickle
from typing import Any

import numpy as np
import shap
import torch

from config import FEATURES_PATH, GRAPH_DATA_PATH, GRAPH_META_PATH, SAVED_MODELS_DIR
from models.gnn_model import load_gnn, predict_node_fraud


class FraudExplainer:
    """Generate human-readable fraud explanations."""

    def __init__(self):
        import joblib

        self.xgb_model = joblib.load(SAVED_MODELS_DIR / "xgboost.joblib")
        with open(SAVED_MODELS_DIR / "feature_names.pkl", "rb") as f:
            self.feature_names: list[str] = pickle.load(f)
        self.gnn_model, _ = load_gnn()
        self.graph_data = torch.load(GRAPH_DATA_PATH, weights_only=False)
        with open(GRAPH_META_PATH, "rb") as f:
            self.graph_meta = pickle.load(f)
        self._shap_explainer = None

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.xgb_model)
        return self._shap_explainer

    def shap_explanation(self, features: np.ndarray) -> dict[str, Any]:
        """SHAP values for XGBoost prediction."""
        features_2d = features.reshape(1, -1)
        shap_values = self.shap_explainer.shap_values(features_2d)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = dict(zip(self.feature_names, shap_values[0].tolist()))
        top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return {"contributions": contributions, "top_features": top_features}

    def gnn_graph_explanation(self, sender: str, receiver: str) -> dict[str, Any]:
        """Graph-based explanation: neighbors, paths, suspicious connections."""
        node_map = self.graph_meta["node_to_idx"]
        idx_to_node = self.graph_meta["idx_to_node"]
        edge_index = self.graph_data.edge_index

        result: dict[str, Any] = {
            "influential_neighbors": [],
            "suspicious_connections": [],
            "transaction_path": [],
        }

        if sender not in node_map or receiver not in node_map:
            return result

        s_idx = node_map[sender]
        r_idx = node_map[receiver]

        with torch.no_grad():
            probs = predict_node_fraud(self.gnn_model, self.graph_data)

        # Outgoing neighbors of sender
        out_neighbors = edge_index[1, edge_index[0] == s_idx].tolist()
        in_neighbors = edge_index[0, edge_index[1] == s_idx].tolist()
        neighbor_idxs = set(out_neighbors + in_neighbors)

        neighbor_scores = []
        for n_idx in neighbor_idxs:
            n_prob = float(probs[n_idx])
            if n_prob > 0.3:
                neighbor_scores.append(
                    {"account": idx_to_node[n_idx], "fraud_probability": round(n_prob, 4)}
                )
        result["influential_neighbors"] = sorted(
            neighbor_scores, key=lambda x: x["fraud_probability"], reverse=True
        )[:5]

        # Suspicious connections (high-risk edges from sender)
        for dst in out_neighbors:
            dst_prob = float(probs[dst])
            if dst_prob > 0.5:
                result["suspicious_connections"].append(
                    {"from": sender, "to": idx_to_node[dst], "risk": round(dst_prob, 4)}
                )

        # Simple 2-hop path sender -> receiver
        result["transaction_path"] = [sender, receiver]
        for mid in out_neighbors:
            mid_edges = edge_index[1, edge_index[0] == mid].tolist()
            if r_idx in mid_edges:
                result["transaction_path"] = [sender, idx_to_node[mid], receiver]
                break

        return result

    def generate_explanation_text(
        self,
        features: np.ndarray,
        sender: str,
        receiver: str,
        fraud_score: float,
    ) -> str:
        """Natural language explanation combining SHAP and graph insights."""
        shap_info = self.shap_explanation(features)
        graph_info = self.gnn_graph_explanation(sender, receiver)

        parts = []

        if fraud_score > 0.5:
            parts.append("Transaction flagged as potentially fraudulent.")
        else:
            parts.append("Transaction appears legitimate based on ensemble analysis.")

        top = shap_info["top_features"]
        if top:
            feat_str = ", ".join(f"{name} ({val:+.3f})" for name, val in top)
            parts.append(f"Key XGBoost drivers: {feat_str}.")

        amount_idx = self.feature_names.index("amount") if "amount" in self.feature_names else None
        if amount_idx is not None and features[amount_idx] > 1.5:
            parts.append("Amount is unusually high relative to typical transactions.")

        if graph_info["influential_neighbors"]:
            risky = graph_info["influential_neighbors"][0]["account"]
            parts.append(f"Sender is connected to previously suspicious account {risky}.")

        if graph_info["suspicious_connections"]:
            conn = graph_info["suspicious_connections"][0]
            parts.append(
                f"Suspicious connection detected: {conn['from']} -> {conn['to']} (risk {conn['risk']})."
            )

        if len(graph_info["transaction_path"]) > 2:
            path = " -> ".join(graph_info["transaction_path"])
            parts.append(f"Multi-hop transaction path identified: {path}.")

        return " ".join(parts)


_explainer: FraudExplainer | None = None


def get_explainer() -> FraudExplainer:
    global _explainer
    if _explainer is None:
        _explainer = FraudExplainer()
    return _explainer
