"""
Fetches the raw dataset from Kaggle and places it at data/raw/creditcard.csv.

Requires the Kaggle CLI to be installed and authenticated
(~/.kaggle/kaggle.json with your API credentials — see
https://www.kaggle.com/docs/api for how to generate one).

Run with:  python -m src.data.fetch_data
"""

import shutil
import subprocess
import zipfile
from pathlib import Path

from src.config import settings

KAGGLE_DATASET = "kanjerih/credit-card-fraud-dataset"


def fetch_data() -> None:
    raw_dir = settings.raw_data_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "credit-card-fraud-dataset.zip"

    print(f"Downloading {KAGGLE_DATASET} via Kaggle CLI...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(raw_dir),
        ],
        check=True,
    )

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    # Kaggle's zip may nest the CSV or name it differently across versions —
    # normalize to the exact path the rest of the pipeline expects.
    candidates = list(raw_dir.glob("*.csv"))
    csv_candidates = [f for f in candidates if f.name != settings.raw_data_path.name]
    if not settings.raw_data_path.exists() and csv_candidates:
        shutil.move(str(csv_candidates[0]), str(settings.raw_data_path))

    if not settings.raw_data_path.exists():
        raise FileNotFoundError(
            f"Expected {settings.raw_data_path} after extraction but it wasn't found. "
            f"Files present: {list(raw_dir.iterdir())}"
        )

    zip_path.unlink(missing_ok=True)
    print(f"✅ Data ready at {settings.raw_data_path}")


if __name__ == "__main__":
    fetch_data()
