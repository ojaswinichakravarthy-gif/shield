

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR,
    FEATURES_PATH,
    METRICS_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SAVED_MODELS_DIR,
)
from fraud_ring_detection import FraudRingDetector
from utils import build_feature_vector, load_metrics

st.set_page_config(page_title="SHIELD-AI Dashboard", page_icon="🛡️", layout="wide")

MODELS_READY = (SAVED_MODELS_DIR / "xgboost.joblib").exists()


def load_data():
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)
    return None


@st.cache_data
def load_engineered():
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, "rb") as f:
            return pickle.load(f)["df"]
    return None


def page_overview():
    st.title("🛡️ SHIELD-AI")
    st.subheader("UPI Fraud Detection — Ensemble Learning + Graph Neural Networks")

    col1, col2, col3, col4 = st.columns(4)
    metrics = load_metrics()
    col1.metric("Precision", f"{metrics.get('precision', 0):.2%}" if metrics else "N/A")
    col2.metric("Recall", f"{metrics.get('recall', 0):.2%}" if metrics else "N/A")
    col3.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}" if metrics else "N/A")
    col4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics else "N/A")

    st.markdown("""
    ### Architecture
    SHIELD-AI combines four ML models with weighted ensemble fusion:

    | Model | Weight |
    |-------|--------|
    | Logistic Regression | 15% |
    | Random Forest | 25% |
    | XGBoost | 30% |
    | GraphSAGE (GNN) | 30% |

    **Threshold:** fraud_probability > 0.5 → Fraud
    """)

    if not MODELS_READY:
        st.warning("Models not trained. Run `python scripts/train.py` first.")


def page_dataset_stats():
    st.header("Dataset Statistics")
    df = load_data()
    if df is None:
        st.error("No dataset found. Run training to generate data.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Rate", f"{df['isFraud'].mean():.2%}")
    c3.metric("Unique Accounts", df["nameOrig"].nunique() if "nameOrig" in df.columns else "N/A")

    st.subheader("Transaction Types")
    st.bar_chart(df["type"].value_counts())

    st.subheader("Amount Distribution")
    st.line_chart(df["amount"].describe())

    st.subheader("Fraud vs Legitimate Amounts")
    fraud_amt = df.groupby("isFraud")["amount"].mean()
    st.bar_chart(fraud_amt)

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)


def page_fraud_prediction():
    st.header("Fraud Prediction")

    if not MODELS_READY:
        st.error("Train models first: python scripts/train.py")
        return

    from ensemble import get_predictor
    from explainability import get_explainer

    eng_df = load_engineered()

    accounts = (
        sorted(
            set(eng_df["sender"].unique())
            | set(eng_df["receiver"].unique())
        )
        if eng_df is not None
        else []
    )

    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        sender = (
            st.selectbox("Sender Account", accounts)
            if accounts
            else st.text_input("Sender")
        )

    with col2:
        receiver = (
            st.selectbox("Receiver Account", accounts)
            if accounts
            else st.text_input("Receiver")
        )

    tx_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    
    if st.button("Predict Fraud Risk", type="primary"):

        features = build_feature_vector(
            sender=sender,
            receiver=receiver,
            amount=amount,
            tx_type=tx_type,
    
        )

        predictor = get_predictor()
        explainer = get_explainer()

        result = predictor.predict_single(
            features,
            sender,
            receiver
        )

        explanation = explainer.generate_explanation_text(
            features,
            sender,
            receiver,
            result["fraud_probability"]
        )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fraud Probability",
            f"{result['fraud_probability']:.2%}"
        )

        c2.metric(
            "Prediction",
            result["prediction"]
        )

        c3.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.info(explanation)

        st.subheader("Per-Model Scores")

        scores_df = pd.DataFrame(
            [result["model_scores"]]
        ).T

        scores_df.columns = ["Probability"]

        st.bar_chart(scores_df)


def page_graph_visualization():
    st.header("Graph Visualization")
    graph_path = DATA_DIR / "transaction_graph.gpickle"
    if not graph_path.exists():
        st.error("Graph not built. Run training first.")
        return

    import pickle as pkl

    with open(graph_path, "rb") as f:
        G = pkl.load(f)

    st.write(f"**Nodes:** {G.number_of_nodes()} | **Edges:** {G.number_of_edges()}")

    max_nodes = st.slider("Max nodes to display", 10, 100, 30)
    sub_nodes = list(G.nodes())[:max_nodes]
    subG = G.subgraph(sub_nodes).copy()

    pos = nx.spring_layout(subG, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(subG, pos, ax=ax, with_labels=True, node_color="lightblue", node_size=500, font_size=8, arrows=True)
    ax.set_title("Transaction Graph (subset)")
    st.pyplot(fig)
    plt.close()


def page_fraud_ring_detection():
    st.header("Fraud Ring Detection")
    if not FEATURES_PATH.exists():
        st.error("Run training first.")
        return

    with st.spinner("Analyzing fraud rings..."):
        detector = FraudRingDetector()
        report = detector.full_report()

    tab1, tab2, tab3 = st.tabs(["Circular Transactions", "Multi-hop Chains", "High-risk Communities"])

    with tab1:
        st.subheader(f"Circular Patterns ({len(report['circular_transactions'])})")
        for i, cycle in enumerate(report["circular_transactions"][:10]):
            st.write(f"{i + 1}. {' → '.join(cycle)} → {cycle[0]}")

    with tab2:
        st.subheader(f"Multi-hop Fraud Chains ({len(report['multi_hop_chains'])})")
        for chain in report["multi_hop_chains"][:10]:
            path = " → ".join(chain["path"])
            st.write(f"**{path}** — {chain['hops']} hops, {chain['fraud_edges']} fraud edges")

    with tab3:
        st.subheader(f"High-risk Communities ({len(report['high_risk_communities'])})")
        for comm in report["high_risk_communities"][:10]:
            st.write(
                f"Community {comm['community_id']}: {comm['size']} members, "
                f"fraud ratio {comm['fraud_ratio']:.2%}, fraud accounts: {comm['fraud_accounts']}"
            )


def page_model_metrics():
    st.header("Model Metrics")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run `python scripts/evaluate.py` after training.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1", f"{metrics['f1_score']:.4f}")

    st.subheader("Per-Model Comparison")
    if "per_model" in metrics:
        per_model_df = pd.DataFrame(metrics["per_model"]).T
        st.dataframe(per_model_df, use_container_width=True)

    roc_path = DATA_DIR / "roc_curve.png"
    cm_path = DATA_DIR / "confusion_matrix.png"

    col1, col2 = st.columns(2)
    with col1:
        if roc_path.exists():
            st.subheader("ROC Curve")
            st.image(str(roc_path))
    with col2:
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_path))

    st.subheader("Classification Report")
    st.code(metrics.get("classification_report", ""))


def page_fusion_analysis():
    st.header("Ensemble Fusion Analysis")
    metrics = load_metrics()
    if not metrics:
        st.warning("Fusion analysis data not found.")
        return

    per_model = pd.DataFrame(
            metrics["per_model"]
        ).T

    st.subheader("Per Model Comparison")

    st.dataframe(
            per_model,
            use_container_width=True
        )

    st.subheader("F1 Score Comparison")

    st.bar_chart(
            per_model["f1_score"]
        )

    st.markdown("""
        ## Why Fusion?

    SHIELD-AI combines:

    - Logistic Regression
    - Random Forest
    - XGBoost
    - Graph Neural Network

    Each model captures different fraud characteristics.

    Fusion improves robustness and reduces dependence on a single model.
    """)

    st.code(
        """
Fraud Score =
0.15 × Logistic Regression
+ 0.25 × Random Forest
+ 0.30 × XGBoost
+ 0.30 × GNN
"""
    )
    roc_compare = DATA_DIR / "roc_comparison.png"
    if roc_compare.exists():
        st.subheader("ROC Comparison")
        st.image(str(roc_compare))


def page_cross_dataset_validation():
    st.header("Cross-Dataset Validation")

    st.markdown("""
    ### Objective

    Evaluate SHIELD-AI concepts on a completely different fraud dataset
    (Credit Card Fraud Dataset - ULB).

    This demonstrates model generalization beyond PaySim.
    """)

    results_path = Path("research/creditcard_model_results.csv")

    if not results_path.exists():
        st.warning("creditcard_model_results.csv not found.")
        return

    df = pd.read_csv(results_path)

    st.subheader("Model Performance")

    st.dataframe(df, use_container_width=True)

    st.subheader("F1 Score Comparison")

    st.bar_chart(
        df.set_index("Model")["F1"]
    )

    st.subheader("Key Observation")

    st.success(
        """
        Random Forest achieved the strongest standalone performance.
        Ensemble remained competitive and demonstrated robustness across domains.
        """
    )


def page_threshold_optimization():
    st.header("Threshold Optimization")

    st.markdown("""
    Fraud detection systems require balancing:

    - Precision
    - Recall

    Threshold tuning allows SHIELD-AI to reduce false positives
    without sacrificing too much recall.
    """)

    threshold_file = Path(
        "research/creditcard_ensemble_results2.csv"
    )

    if not threshold_file.exists():
        st.warning("Threshold results not found.")
        return

    df = pd.read_csv(threshold_file)

    st.dataframe(df.head())

    st.subheader("F1 vs Threshold")

    st.line_chart(
        df.set_index("threshold")["f1"]
    )

    best_row = df.loc[df["f1"].idxmax()]

    st.success(
        f"""
        Best Threshold = {best_row['threshold']:.2f}

        Precision = {best_row['precision']:.3f}

        Recall = {best_row['recall']:.3f}

        F1 = {best_row['f1']:.3f}
        """
    )


PAGES = {
    "Overview": page_overview,
    "Dataset Statistics": page_dataset_stats,
    "Fraud Prediction": page_fraud_prediction,
    "Graph Visualization": page_graph_visualization,
    "Fraud Ring Detection": page_fraud_ring_detection,
    "Model Metrics": page_model_metrics,
    "Fusion Analysis": page_fusion_analysis,
    "Cross Dataset Validation": page_cross_dataset_validation,
    "Threshold Optimization": page_threshold_optimization,
}
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()

