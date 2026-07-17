"""
API tests. /predict is tested against a small stub model trained
in-memory so the test doesn't depend on a real model.joblib being
present or on the full dataset.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.config import settings


@pytest.fixture(autouse=True)
def stub_model(tmp_path, monkeypatch):
    """Train a tiny stub model and point settings.model_dir at it so
    the API's lifespan loader picks it up."""
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(0, 1, size=(100, len(feature_names))), columns=feature_names)
    X["Amount"] = np.abs(X["Amount"])
    y = pd.Series(rng.randint(0, 2, size=100))

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)

    monkeypatch.setattr(settings, "model_dir", tmp_path)
    joblib.dump(model, tmp_path / "model.joblib")
    yield


@pytest.fixture
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def _sample_transaction():
    payload = {"Time": 1000.0, "Amount": 50.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=_sample_transaction())
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["threshold_used"] == settings.classification_threshold


def test_predict_rejects_missing_fields(client):
    incomplete = {"Time": 1000.0, "Amount": 50.0}  # missing V1-V28
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_negative_amount(client):
    payload = _sample_transaction()
    payload["Amount"] = -5.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
