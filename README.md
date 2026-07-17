# Credit Card Fraud Detection — MLOps Pipeline

An end-to-end MLOps project: a Random Forest fraud classifier taken from
notebook prototype to a tested, tracked, containerized, CI/CD-deployed
API — with model-quality gates, drift monitoring, and a business-impact
(cost/savings) analysis layered on top of standard classification metrics.

## Why this project exists

Most fraud-detection notebooks stop at precision/recall. This project
asks the next questions: *Is that metric trustworthy across data splits?
What does it cost the business in dollars? How do we stop a worse model
from silently replacing a better one? How do we know when the live data
has drifted from what the model was trained on?* — and answers each with
a concrete piece of the pipeline below.

## Architecture

```
                 ┌─────────────┐
   data/raw/ ──▶ │   DVC       │  data versioning (S3/local remote)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐      ┌──────────────┐
                 │  src/models/│ ───▶ │   MLflow     │  experiment tracking
                 │   train.py  │      │  registry    │  + model registry
                 └──────┬──────┘      └──────────────┘
                        │  (fails build if metrics < gate)
                        ▼
                 ┌─────────────┐
                 │ GitHub      │  lint → test → train-gate → build → deploy
                 │ Actions     │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐      ┌──────────────┐
                 │   Docker    │ ───▶ │    Render    │  hosted API
                 │  container  │      │  (free tier) │
                 └─────────────┘      └──────┬───────┘
                                              ▼
                                       ┌──────────────┐
                                       │  Evidently   │  drift monitoring
                                       │  drift report│
                                       └──────────────┘
```

## Tool stack & why each piece is there

| Layer | Tool | Purpose |
|---|---|---|
| Data ingestion | **Kaggle CLI** | `src/data/fetch_data.py` scripts the download so the raw data is fetched reproducibly, not by hand |
| Data versioning | **DVC** | Reproducible data snapshots without bloating Git; `dvc.yaml` defines the fetch/train/evaluate pipeline as a DAG |
| Experiment tracking | **MLflow** | Every training run logs params, metrics, and the model artifact; a **model registry quality gate** blocks a model from being registered unless it beats minimum recall/precision/MCC thresholds (`src/config.py`) |
| Modeling | **scikit-learn** | RandomForestClassifier, stratified split, 5-fold stratified CV |
| API | **FastAPI + Pydantic** | Typed request/response schemas, input validation, `/health` + `/predict` |
| Testing | **pytest** | Unit tests for data validation, training logic, and the API (all run on synthetic data — no dependency on the real dataset in CI) |
| Code quality | **ruff + black + pre-commit** | Lint/format enforced both locally (pre-commit hook) and in CI |
| Containerization | **Docker** | Single reproducible image for local dev and production |
| Local orchestration | **Docker Compose** | Spins up the API + a local MLflow tracking server together |
| CI/CD | **GitHub Actions** | `ci.yml`: lint → test → train-and-gate on every PR. `cd.yml`: rebuild + push image + deploy on merge to `main` |
| Deployment | **Render** | Free-tier container hosting, deployed via `render.yaml` blueprint |
| Monitoring | **Evidently AI** | Data drift report comparing reference vs. current feature distributions |

## Repo layout

```
├── src/
│   ├── config.py          # single source of truth for paths, hyperparams, thresholds, quality gate
│   ├── data/load_data.py  # schema-validated data loading
│   ├── models/train.py    # stratified split + CV + MLflow logging + registry gate
│   ├── models/evaluate.py # threshold sweep + cost/savings report
│   ├── monitoring/drift_check.py  # Evidently drift report
│   └── api/                # FastAPI serving layer
├── tests/                  # pytest suite (data, model, API)
├── notebooks/               # exploratory work only — not imported by src/
├── .github/workflows/       # ci.yml, cd.yml
├── Dockerfile / docker-compose.yml / render.yaml
├── dvc.yaml                 # DVC pipeline stages
└── Makefile                 # `make train`, `make test`, `make serve`, etc.
```

## Getting started

```bash
git clone <this-repo>
cd fraud-detection-mlops
make install-dev          # installs deps + sets up pre-commit hooks

# Get the data — requires Kaggle API credentials at ~/.kaggle/kaggle.json
# (see https://www.kaggle.com/docs/api). Downloads kanjerih/credit-card-fraud-dataset
# and places it at data/raw/creditcard.csv.
make fetch-data
# Alternatively, once a DVC remote is configured for this repo: dvc pull

make train                 # trains model, logs to MLflow, applies quality gate
make evaluate               # writes reports/threshold_savings_report.json
make test                   # runs the pytest suite
make serve                   # runs the API locally at http://localhost:8000
```

To run the API + MLflow tracking server together:

```bash
make compose-up
```

## Model quality gate

`src/models/train.py` only registers a model in MLflow if it clears the
minimum thresholds defined in `src/config.py`:

```python
min_recall: float = 0.70
min_precision: float = 0.85
min_mcc: float = 0.75
```

If a training run fails the gate, `train.py` exits non-zero — this is
what allows the CI pipeline to **block a regression from being deployed**,
rather than silently shipping a worse model.

## Business-impact analysis

Rather than stopping at precision/recall, `src/models/evaluate.py` sweeps
decision thresholds and computes **net savings** per threshold:

```
net_savings = (fraud $ caught) − (fraud $ missed) − (false positives × review cost)
```

This answers the more useful question for a stakeholder: *not* "how
accurate is the model," but "what does deploying this model at this
threshold actually save us." See `reports/threshold_savings_report.json`
after running `make evaluate`.

## Known limitations / next steps

- The Evidently drift check currently compares two halves of the same
  static dataset as a working example — in production this should run
  on a schedule against freshly logged inference inputs.
- The false-positive cost ($10/review) is a placeholder assumption; a
  sensitivity analysis across a range of costs would make the
  recommended threshold more defensible.
- No feature store — features are computed inline at request time,
  which is fine at this scale but wouldn't scale to a system with
  complex, stateful features (e.g. rolling transaction counts).
- Model retraining is manual (`make train`); a scheduled retraining
  trigger (e.g. GitHub Actions cron, gated by the drift check) would
  close the loop into a fully automated MLOps cycle.
