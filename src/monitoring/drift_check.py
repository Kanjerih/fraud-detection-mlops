"""
Data drift monitoring using Evidently AI.

Compares a "reference" window (the training distribution) against a
"current" window (newer production/incoming data) and flags drift in
the input features. In production this would run on a schedule (e.g.
a daily GitHub Actions cron, or a Render cron job) against freshly
logged inference inputs; here it's wired to run against two slices of
the same dataset as a working example.

Run with:  python -m src.monitoring.drift_check
"""

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.config import settings
from src.data.load_data import load_raw_data


def run_drift_check(reference_df, current_df, output_path):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(output_path))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    return drift_detected


def main():
    df = load_raw_data()

    # Example split: first half as "reference" (training-time distribution),
    # second half as "current" (stand-in for freshly arriving data).
    # In production, swap `current` for a logged table of recent live inputs.
    midpoint = len(df) // 2
    reference_df = df.iloc[:midpoint].drop(columns=["Class"])
    current_df = df.iloc[midpoint:].drop(columns=["Class"])

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.reports_dir / "drift_report.html"

    drift_detected = run_drift_check(reference_df, current_df, output_path)

    print(f"Drift report written to {output_path}")
    if drift_detected:
        print("⚠️  Data drift detected — consider retraining.")
    else:
        print("✅ No significant drift detected.")


if __name__ == "__main__":
    main()
