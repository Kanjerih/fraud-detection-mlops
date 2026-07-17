"""
Data loading and validation utilities.

Keeping this separate from training means the same validated load path
is used by training, tests, and (if needed) batch-scoring jobs.
"""

import pandas as pd

from src.config import settings

EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw credit card transactions CSV and validate its schema."""
    path = path or settings.raw_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run `dvc pull` or place creditcard.csv in data/raw/."
        )

    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the dataframe doesn't match the expected fraud dataset schema."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Unexpected nulls found in columns: {null_cols}")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class column must be binary (0 = valid, 1 = fraud).")


def split_features_target(df: pd.DataFrame):
    """Split into feature matrix X and target vector y."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y
