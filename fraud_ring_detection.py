"""
Phase 8: Fraud Ring Detection — cycles, multi-hop chains, high-risk communities.
"""

import pickle
from typing import Any

import networkx as nx
import numpy as np

from config import FEATURES_PATH, TARGET_COLUMN


class FraudRingDetector:
    """Detect fraud rings using graph topology and community analysis."""

    def __init__(self, G: nx.DiGraph | None = None):
        if G is None:
            G = self._load_graph()
        self.G = G

    @staticmethod
    def _load_graph() -> nx.DiGraph:
        from graph_builder import build_transaction_graph

        with open(FEATURES_PATH, "rb") as f:
            df = pickle.load(f)["df"]
        return build_transaction_graph(df)

    def detect_circular_transactions(self, max_length: int = 6) -> list[list[str]]:
        """
        Detect circular transaction patterns: A -> B -> C -> A.
        Uses simple_cycles on the underlying undirected view for efficiency.
        """
        undirected = self.G.to_undirected()
        cycles = []
        try:
            for cycle in nx.simple_cycles(self.G):
                if 3 <= len(cycle) <= max_length:
                    cycles.append(cycle)
                    if len(cycles) >= 100:
                        break
        except nx.NetworkXError:
            pass

        if not cycles:
            # Fallback: find cycles in undirected graph
            for component in nx.connected_components(undirected):
                sub = undirected.subgraph(component)
                if sub.number_of_nodes() < 3:
                    continue
                try:
                    cycle_basis = nx.cycle_basis(sub)
                    for cyc in cycle_basis:
                        if len(cyc) >= 3:
                            cycles.append(cyc)
                            if len(cycles) >= 100:
                                return cycles
                except Exception:
                    continue
        return cycles[:50]

    def detect_multi_hop_fraud(self, min_hops: int = 3, max_hops: int = 5) -> list[dict[str, Any]]:
        """
        Detect multi-hop fraud chains: A -> B -> C -> D.
        Prioritizes paths involving accounts with fraud history on edges.
        """
        fraud_edges = {
            (u, v)
            for u, v, d in self.G.edges(data=True)
            if d.get("fraud_count", 0) > 0
        }
        fraud_nodes = {u for u, v in fraud_edges} | {v for u, v in fraud_edges}

        chains = []
        for source in list(fraud_nodes)[:200]:
            for target in self.G.nodes():
                if source == target:
                    continue
                try:
                    for path in nx.all_simple_paths(
                        self.G, source, target, cutoff=max_hops
                    ):
                        if min_hops <= len(path) - 1 <= max_hops:
                            fraud_on_path = sum(
                                1 for i in range(len(path) - 1) if (path[i], path[i + 1]) in fraud_edges
                            )
                            if fraud_on_path > 0:
                                chains.append(
                                    {
                                        "path": path,
                                        "hops": len(path) - 1,
                                        "fraud_edges": fraud_on_path,
                                    }
                                )
                            if len(chains) >= 50:
                                return sorted(chains, key=lambda x: x["fraud_edges"], reverse=True)
                except nx.NetworkXError:
                    continue
        return sorted(chains, key=lambda x: x["fraud_edges"], reverse=True)[:50]

    def detect_high_risk_communities(self) -> list[dict[str, Any]]:
        """Community detection with fraud density scoring."""
        undirected = self.G.to_undirected()
        if undirected.number_of_nodes() < 3:
            return []

        try:
            communities = list(nx.community.greedy_modularity_communities(undirected))
        except Exception:
            communities = [set(undirected.nodes())]

        results = []
        with open(FEATURES_PATH, "rb") as f:
            df = pickle.load(f)["df"]

        fraud_accounts = set()
        fraud_df = df[df[TARGET_COLUMN] == 1]
        for _, row in fraud_df.iterrows():
            fraud_accounts.add(str(row["sender"]))
            fraud_accounts.add(str(row["receiver"]))

        for i, community in enumerate(communities):
            members = list(community)
            fraud_count = sum(1 for m in members if m in fraud_accounts)
            fraud_ratio = fraud_count / max(len(members), 1)
            if fraud_ratio > 0.05 or fraud_count >= 2:
                results.append(
                    {
                        "community_id": i,
                        "size": len(members),
                        "fraud_accounts": fraud_count,
                        "fraud_ratio": round(fraud_ratio, 4),
                        "members": members[:20],
                    }
                )

        return sorted(results, key=lambda x: x["fraud_ratio"], reverse=True)[:20]

    def full_report(self) -> dict[str, Any]:
        """Run all fraud ring detection methods."""
        return {
            "circular_transactions": self.detect_circular_transactions(),
            "multi_hop_chains": self.detect_multi_hop_fraud(),
            "high_risk_communities": self.detect_high_risk_communities(),
        }


if __name__ == "__main__":
    detector = FraudRingDetector()
    report = detector.full_report()
    print(f"Circular patterns: {len(report['circular_transactions'])}")
    print(f"Multi-hop chains: {len(report['multi_hop_chains'])}")
    print(f"High-risk communities: {len(report['high_risk_communities'])}")
