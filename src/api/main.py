"""
FastAPI serving layer for the fraud detection model.

Loads the trained model once at startup (not per-request) and exposes:
  GET  /health   - liveness + model status, used by Render's health check
  POST /predict  - score a single transaction

Run locally with:  uvicorn src.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

model_state = {"model": None, "version": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = settings.model_dir / "model.joblib"
    try:
        model_state["model"] = joblib.load(model_path)
        model_state["version"] = model_path.stat().st_mtime.__str__()
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.warning(
            f"No model found at {model_path}. "
            "/predict will return 503 until a model is trained and present."
        )
    yield
    model_state["model"] = None


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves fraud predictions from a RandomForest model "
    "trained with a stratified split, validated via cross-validation, "
    "and threshold-tuned against a cost-of-error analysis.",
    version="1.0.0",
    lifespan=lifespan,
)
# CORS: allows the separately-hosted frontend (a different origin) to call
# this API from a browser. Wide open (*) is fine for a portfolio demo with
# no auth/sensitive data; for a real production system, replace with an
# explicit list of allowed frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_state["model"] is not None,
        model_version=model_state["version"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train a model first."
        )

    row = pd.DataFrame([transaction.model_dump()])
    prob = float(model.predict_proba(row)[:, 1][0])
    is_fraud = prob >= settings.classification_threshold

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 6),
        threshold_used=settings.classification_threshold,
        model_version=model_state["version"],
    )
