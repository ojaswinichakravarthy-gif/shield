"""
Phase 3: Graph Construction using NetworkX and PyTorch Geometric export.

Nodes: Accounts | Edges: Transactions (sender -> receiver)
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import (
    ACCOUNT_MAP_PATH,
    FEATURES_PATH,
    GRAPH_DATA_PATH,
    GRAPH_META_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)


def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph with accounts as nodes and transactions as edges."""
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = str(row["sender"])
        receiver = str(row["receiver"])
        amount = float(row["amount"])
        is_fraud = int(row.get(TARGET_COLUMN, 0))

        if sender not in G:
            G.add_node(sender)
        if receiver not in G:
            G.add_node(receiver)

        if G.has_edge(sender, receiver):
            G[sender][receiver]["weight"] += 1
            G[sender][receiver]["total_amount"] += amount
            G[sender][receiver]["fraud_count"] += is_fraud
        else:
            G.add_edge(
                sender,
                receiver,
                weight=1,
                total_amount=amount,
                fraud_count=is_fraud,
            )

    return G


def compute_node_features(G: nx.DiGraph, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute per-node features: degree, tx count, avg amount, fraud history."""
    fraud_by_account: dict[str, int] = {}
    for _, row in df.iterrows():
        for acc in (str(row["sender"]), str(row["receiver"])):
            fraud_by_account[acc] = fraud_by_account.get(acc, 0) + int(row.get(TARGET_COLUMN, 0))

    tx_count = {}
    avg_amount = {}
    for _, row in df.iterrows():
        s = str(row["sender"])
        tx_count[s] = tx_count.get(s, 0) + 1
        if s not in avg_amount:
            avg_amount[s] = []
        avg_amount[s].append(float(row["amount"]))

    features = {}
    for node in G.nodes():
        out_deg = G.out_degree(node)
        in_deg = G.in_degree(node)
        degree = out_deg + in_deg
        count = tx_count.get(node, 0)
        amounts = avg_amount.get(node, [0.0])
        fraud_hist = fraud_by_account.get(node, 0)
        features[node] = np.array(
            [degree, count, np.mean(amounts), fraud_hist],
            dtype=np.float32,
        )
    return features


def graph_to_pyg(G: nx.DiGraph, node_features: dict[str, np.ndarray], df: pd.DataFrame) -> tuple[Data, dict]:
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Node labels: 1 if account involved in any fraud transaction
    fraud_label = {}
    for _, row in df.iterrows():
        if int(row.get(TARGET_COLUMN, 0)) == 1:
            fraud_label[str(row["sender"])] = 1
            fraud_label[str(row["receiver"])] = 1
    y = torch.tensor([fraud_label.get(n, 0) for n in nodes], dtype=torch.float)

    x = torch.tensor(np.stack([node_features[n] for n in nodes]), dtype=torch.float)

    edge_index = []
    edge_attr = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_attr.append([data.get("weight", 1), data.get("total_amount", 0)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Normalize node features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    meta = {"node_to_idx": node_to_idx, "idx_to_node": {i: n for n, i in node_to_idx.items()}, "nodes": nodes}
    return data, meta


def build_and_save(df: pd.DataFrame | None = None) -> tuple[Data, dict]:
    """Full graph pipeline: build, feature extraction, export."""
    if df is None:
        with open(FEATURES_PATH, "rb") as f:
            df = pickle.load(f)["df"]

    G = build_transaction_graph(df)
    node_features = compute_node_features(G, df)
    pyg_data, meta = graph_to_pyg(G, node_features, df)

    GRAPH_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pyg_data, GRAPH_DATA_PATH)

    with open(GRAPH_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    with open(ACCOUNT_MAP_PATH, "wb") as f:
        pickle.dump(meta["node_to_idx"], f)

    nx_graph_path = GRAPH_DATA_PATH.parent / "transaction_graph.gpickle"
    with open(nx_graph_path, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph saved: {len(G.nodes())} nodes, {len(G.edges())} edges -> {GRAPH_DATA_PATH}")
    return pyg_data, meta


if __name__ == "__main__":
    from feature_engineering import engineer_features
    from preprocessing import preprocess

    if not PROCESSED_DATA_PATH.exists():
        preprocess()
    engineer_features()
    build_and_save()
