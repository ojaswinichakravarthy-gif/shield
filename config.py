"""SHIELD-AI configuration and shared paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

RAW_DATA_PATH = DATA_DIR / "paysim.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"
FEATURES_PATH = DATA_DIR / "engineered_features.pkl"
GRAPH_DATA_PATH = DATA_DIR / "graph_data.pt"
GRAPH_META_PATH = DATA_DIR / "graph_meta.pkl"
METRICS_PATH = DATA_DIR / "metrics.json"
ACCOUNT_MAP_PATH = DATA_DIR / "account_map.pkl"

ENSEMBLE_WEIGHTS = {
    "logistic_regression": 0.15,
    "random_forest": 0.25,
    "xgboost": 0.30,
    "gnn": 0.30,
}

FRAUD_THRESHOLD = 0.5

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_encoded",
    "balance_difference_sender",
    "balance_difference_receiver",
    "transaction_frequency",
    "average_transaction_amount",
    "sender_out_degree",
    "receiver_in_degree",
]

TARGET_COLUMN = "isFraud"

RANDOM_STATE = 42
TEST_SIZE = 0.2

GNN_HIDDEN_DIM = 64
GNN_DROPOUT = 0.3
GNN_EPOCHS = 50
GNN_LR = 0.01
