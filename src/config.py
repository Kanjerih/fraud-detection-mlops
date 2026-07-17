"""
Centralized configuration for the fraud detection pipeline.

All paths, hyperparameters, and thresholds are defined here so that
training, evaluation, and serving code share a single source of truth.
Values can be overridden via environment variables (see .env.example).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_data_path: Path = project_root / "data" / "raw" / "creditcard.csv"
    processed_dir: Path = project_root / "data" / "processed"
    model_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"

    # --- MLflow ---
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "credit-card-fraud-detection"
    registered_model_name: str = "fraud-detection-rf"

    # --- Train/test split ---
    test_size: float = 0.2
    random_state: int = 42
    n_cv_folds: int = 5

    # --- Model ---
    n_estimators: int = 100

    # --- Decision threshold (tuned via cost analysis, see reports/) ---
    classification_threshold: float = 0.3

    # --- Cost assumptions for savings analysis ---
    false_positive_cost: float = 10.0

    # --- Minimum acceptable metrics (CI training-validation gate) ---
    min_recall: float = 0.70
    min_precision: float = 0.85
    min_mcc: float = 0.75


settings = Settings()
