"""
Phase 9: FastAPI Backend for SHIELD-AI fraud detection API.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explainability import get_explainer
from ensemble import get_predictor
from utils import build_feature_vector


class TransactionRequest(BaseModel):
    sender: str = Field(..., example="C0001")
    receiver: str = Field(..., example="C0137")
    amount: float = Field(..., gt=0, example=50000.0)
    type: str = Field(..., example="TRANSFER")
    oldbalanceOrg: float | None = None
    newbalanceOrig: float | None = None
    oldbalanceDest: float | None = None
    newbalanceDest: float | None = None


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    explanation: str
    risk_level: str
    model_scores: dict[str, float] | None = None


def _models_available() -> bool:
    return (Path(__file__).parent / "models" / "saved" / "xgboost.joblib").exists()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _models_available():
        get_predictor()
        get_explainer()
    yield


app = FastAPI(
    title="SHIELD-AI",
    description="UPI Fraud Detection using Ensemble Learning + Graph Neural Networks",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "SHIELD-AI",
        "status": "running",
        "models_loaded": _models_available(),
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "models_ready": _models_available()}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """
    Predict fraud risk for a UPI transaction.

    Example:
        POST /predict
        {"sender": "C0001", "receiver": "C0137", "amount": 50000, "type": "TRANSFER"}
    """
    if not _models_available():
        raise HTTPException(
            status_code=503,
            detail="Models not trained. Run: python scripts/train.py",
        )

    try:
        features = build_feature_vector(
            sender=transaction.sender,
            receiver=transaction.receiver,
            amount=transaction.amount,
            tx_type=transaction.type,
            oldbalance_org=transaction.oldbalanceOrg,
            newbalance_orig=transaction.newbalanceOrig,
            oldbalance_dest=transaction.oldbalanceDest,
            newbalance_dest=transaction.newbalanceDest,
        )
        predictor = get_predictor()
        explainer = get_explainer()

        result = predictor.predict_single(features, transaction.sender, transaction.receiver)
        explanation = explainer.generate_explanation_text(
            features, transaction.sender, transaction.receiver, result["fraud_probability"]
        )

        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            prediction=result["prediction"],
            explanation=explanation,
            risk_level=result["risk_level"],
            model_scores=result["model_scores"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
