"""
Evaluation + business-impact reporting.

Loads the currently registered model (or a local model.joblib fallback),
sweeps decision thresholds, and writes a savings report to reports/.
This is what turns raw classification metrics into a dollar figure a
non-technical stakeholder can act on.

Run with:  python -m src.models.evaluate
"""

import json

import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.config import settings
from src.data.load_data import load_raw_data, split_features_target


def sweep_thresholds(X, y, thresholds, n_splits, random_state, fp_cost):
    from sklearn.ensemble import RandomForestClassifier

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {t: {"savings": [], "precision": [], "recall": []} for t in thresholds}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        amounts_val = X_val["Amount"].values

        model = RandomForestClassifier(
            n_estimators=settings.n_estimators, random_state=random_state
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]

        for t in thresholds:
            preds = (probs >= t).astype(int)
            y_val_arr = y_val.values

            tp_mask = (y_val_arr == 1) & (preds == 1)
            fn_mask = (y_val_arr == 1) & (preds == 0)
            fp_mask = (y_val_arr == 0) & (preds == 1)

            net_savings = (
                amounts_val[tp_mask].sum()
                - amounts_val[fn_mask].sum()
                - fp_mask.sum() * fp_cost
            )
            results[t]["savings"].append(net_savings)
            results[t]["precision"].append(
                precision_score(y_val_arr, preds, zero_division=0)
            )
            results[t]["recall"].append(recall_score(y_val_arr, preds, zero_division=0))

    summary = {}
    for t in thresholds:
        summary[str(t)] = {
            "net_savings_mean": float(np.mean(results[t]["savings"])),
            "net_savings_std": float(np.std(results[t]["savings"])),
            "precision_mean": float(np.mean(results[t]["precision"])),
            "recall_mean": float(np.mean(results[t]["recall"])),
        }
    return summary


def main():
    df = load_raw_data()
    X, y = split_features_target(df)

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    summary = sweep_thresholds(
        X,
        y,
        thresholds,
        settings.n_cv_folds,
        settings.random_state,
        settings.false_positive_cost,
    )

    best_threshold = max(summary, key=lambda t: summary[t]["net_savings_mean"])

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / "threshold_savings_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {"results": summary, "recommended_threshold": best_threshold}, f, indent=2
        )

    print(f"Report written to {report_path}")
    print(f"Recommended threshold: {best_threshold} "
          f"(net savings ${summary[best_threshold]['net_savings_mean']:,.2f})")


if __name__ == "__main__":
    main()
