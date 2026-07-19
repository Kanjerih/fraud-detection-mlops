"""
Training entry point for the fraud detection model.

Pipeline:
  1. Load + validate data
  2. Stratified train/test split
  3. Train RandomForestClassifier
  4. 5-fold stratified cross-validation (robustness check)
  5. Log params/metrics/artifacts to MLflow, register model if it clears
     the minimum-metric gate defined in config (used by CI to block bad
     models from being promoted).

Run with:  python -m src.models.train
"""

import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import settings
from src.data.load_data import load_raw_data, split_features_target


def cross_validate(X, y, n_splits, random_state) -> dict:
    """5-fold stratified CV, returns mean/std for each metric."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = {"precision": [], "recall": [], "f1": [], "mcc": []}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=settings.n_estimators, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        scores["precision"].append(precision_score(y_val, preds, zero_division=0))
        scores["recall"].append(recall_score(y_val, preds, zero_division=0))
        scores["f1"].append(f1_score(y_val, preds, zero_division=0))
        scores["mcc"].append(matthews_corrcoef(y_val, preds))

    return {f"cv_{metric}_mean": float(np.mean(vals)) for metric, vals in scores.items()} | {
        f"cv_{metric}_std": float(np.std(vals)) for metric, vals in scores.items()
    }


def train() -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = load_raw_data()
    X, y = split_features_target(df)

    xtrain, xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": settings.n_estimators,
                "test_size": settings.test_size,
                "random_state": settings.random_state,
                "classification_threshold": settings.classification_threshold,
            }
        )

        model = RandomForestClassifier(
            n_estimators=settings.n_estimators, random_state=settings.random_state, n_jobs=-1
        )
        model.fit(xtrain, ytrain)

        probs = model.predict_proba(xtest)[:, 1]
        preds = (probs >= settings.classification_threshold).astype(int)

        holdout_metrics = {
            "holdout_accuracy": accuracy_score(ytest, preds),
            "holdout_precision": precision_score(ytest, preds, zero_division=0),
            "holdout_recall": recall_score(ytest, preds, zero_division=0),
            "holdout_f1": f1_score(ytest, preds, zero_division=0),
            "holdout_mcc": matthews_corrcoef(ytest, preds),
        }
        mlflow.log_metrics(holdout_metrics)

        cv_metrics = cross_validate(X, y, settings.n_cv_folds, settings.random_state)
        mlflow.log_metrics(cv_metrics)

        settings.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.model_dir / "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        signature = mlflow.models.infer_signature(xtrain, model.predict(xtrain))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=None,  # registered conditionally below
        )

        all_metrics = {**holdout_metrics, **cv_metrics}
        print("=== Training run complete ===")
        for k, v in all_metrics.items():
            print(f"{k}: {v:.4f}")

        gate_passed = (
            cv_metrics["cv_recall_mean"] >= settings.min_recall
            and cv_metrics["cv_precision_mean"] >= settings.min_precision
            and cv_metrics["cv_mcc_mean"] >= settings.min_mcc
        )
        mlflow.set_tag("gate_passed", str(gate_passed))

        if gate_passed:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", settings.registered_model_name)
            print(
                f"\n✅ Model passed quality gate and was registered "
                f"as '{settings.registered_model_name}'."
            )
        else:
            print(
                "\n❌ Model FAILED quality gate "
                f"(min_recall={settings.min_recall}, "
                f"min_precision={settings.min_precision}, "
                f"min_mcc={settings.min_mcc}). Not registered."
            )

        return {"metrics": all_metrics, "gate_passed": gate_passed}


if __name__ == "__main__":
    result = train()
    # Non-zero exit if the model fails the quality gate — this is what
    # lets the CI pipeline block a regression from being deployed.
    sys.exit(0 if result["gate_passed"] else 1)
