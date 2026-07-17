"""Request/response schemas for the fraud detection API."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A single transaction to score. V1-V28 are the PCA-transformed
    features from the original dataset; Time and Amount are raw."""

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
