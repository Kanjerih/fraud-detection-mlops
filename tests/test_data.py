"""Tests for data loading and schema validation."""

import pandas as pd
import pytest

from src.data.load_data import split_features_target, validate_schema


def _make_valid_df(n=10):
    data = {"Time": range(n), "Amount": [10.0] * n, "Class": [0] * (n - 1) + [1]}
    for i in range(1, 29):
        data[f"V{i}"] = [0.1] * n
    return pd.DataFrame(data)


def test_validate_schema_accepts_valid_df():
    df = _make_valid_df()
    validate_schema(df)  # should not raise


def test_validate_schema_rejects_missing_columns():
    df = _make_valid_df().drop(columns=["V1"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(df)


def test_validate_schema_rejects_nulls():
    df = _make_valid_df()
    df.loc[0, "Amount"] = None
    with pytest.raises(ValueError, match="nulls"):
        validate_schema(df)


def test_validate_schema_rejects_non_binary_class():
    df = _make_valid_df()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_schema(df)


def test_split_features_target():
    df = _make_valid_df()
    X, y = split_features_target(df)
    assert "Class" not in X.columns
    assert y.name == "Class"
    assert len(X) == len(y) == len(df)
