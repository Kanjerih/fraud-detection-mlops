"""
Tests for the training pipeline.

Uses a small synthetic, separable dataset rather than the full 284k-row
real dataset so the test suite runs in seconds and doesn't require the
data file to be present in CI (data is DVC-tracked, not committed).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.train import cross_validate


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(0)
    n_normal, n_fraud = 500, 50

    normal = pd.DataFrame(rng.normal(0, 1, size=(n_normal, 5)), columns=[f"f{i}" for i in range(5)])
    normal["Class"] = 0

    fraud = pd.DataFrame(rng.normal(4, 1, size=(n_fraud, 5)), columns=[f"f{i}" for i in range(5)])
    fraud["Class"] = 1

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=0)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def test_cross_validate_returns_expected_keys(synthetic_data):
    X, y = synthetic_data
    results = cross_validate(X, y, n_splits=3, random_state=0)

    expected_keys = {
        "cv_precision_mean", "cv_precision_std",
        "cv_recall_mean", "cv_recall_std",
        "cv_f1_mean", "cv_f1_std",
        "cv_mcc_mean", "cv_mcc_std",
    }
    assert expected_keys.issubset(results.keys())


def test_cross_validate_on_separable_data_scores_well(synthetic_data):
    """On a cleanly separable synthetic dataset, the model should score
    highly — this is a smoke test that the CV loop itself isn't broken
    (e.g. label leakage, wrong axis, etc.), not a claim about real-world
    performance."""
    X, y = synthetic_data
    results = cross_validate(X, y, n_splits=3, random_state=0)

    assert results["cv_recall_mean"] > 0.8
    assert results["cv_precision_mean"] > 0.8
    assert results["cv_mcc_mean"] > 0.7
