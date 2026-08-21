
# IT Service Ticket Classification & Confidence-Based Routing

An NLP project that sorts IT service desk tickets into eight support categories. Instead of forcing a prediction on every ticket, the system flags ambiguous cases for manual review based on model confidence.

I started this project to compare a classical TF-IDF + Logistic Regression baseline against a PyTorch TextCNN. The Logistic Regression model ended up performing slightly better on the test set, so that's what runs in the FastAPI backend by default.

## Overview

Most classification tutorials just try to maximize accuracy. In a real helpdesk, you can't afford to auto-route ambiguous tickets. This project focuses on the practical trade-off between coverage (how many tickets we automate) and accuracy (how often the automated routing is correct).

Key details:
- Trained on a public dataset of **47,837 tickets**, stratified into a 70/10/10/10 split (train/tune/calibration/test).
- The routing threshold is tuned strictly on the validation set, keeping the test set isolated.
- The API returns the top 3 categories, the confidence score, and a boolean flag indicating if the ticket needs manual review.

## Results

### Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **TF-IDF + Logistic Regression** | **85.31%** | **85.28%** | **85.34%** |
| TextCNN | 84.88% | 85.01% | 84.94% |

![Model performance comparison](reports/figures/model_comparison.png)

I picked Logistic Regression for the final pipeline because it yielded the best routing coverage on the validation set (while holding auto-routed accuracy above 90%). 

The production baseline uses sigmoid probability calibration through `CalibratedClassifierCV` with 5-fold cross-validation on the training set. On the validation set, it achieved an Expected Calibration Error (ECE) of 0.0866 and a top-label Brier score of 0.1037.

### Confidence-based routing

Threshold selection uses a statistically rigorous procedure: for each candidate threshold on a 0.01-step grid, the script computes a one-sided **exact Clopper-Pearson confidence lower bound** on auto-route accuracy, with a **Bonferroni correction** for the number of candidates evaluated. A threshold is eligible only if its simultaneous lower bound is ≥ 90%. Among eligible thresholds, the one with the highest coverage (most tickets auto-routed) is selected.

For Logistic Regression, the chosen threshold was **0.54**.

| Test-set routing metric | Result |
|---|---:|
| Overall accuracy | 85.31% |
| Auto-route coverage | **86.62%** |
| Accuracy on auto-routed tickets | **90.57%** |
| Manual-review rate | 13.38% |
| Auto-routed tickets | 6,216 / 7,176 |

![Routing accuracy versus coverage](reports/figures/baseline_threshold_tradeoff.png)

On the held-out test set, auto-routed tickets achieved 90.57% accuracy. This figure is a point estimate on the test set; the statistical accuracy guarantee (simultaneous Clopper-Pearson lower bound ≥ 90%) was established on the tune set and should not be re-interpreted as a guarantee on unseen future data.

This converts the classifier into a simple human-in-the-loop routing system: high-confidence tickets are routed automatically, while lower-confidence cases are flagged for manual review.

## Dataset

The project uses the [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset).

Expected columns:

- `Document` — service ticket text
- `Topic_group` — target support category

Categories:

- Access
- Administrative rights
- Hardware
- HR Support
- Internal Project
- Miscellaneous
- Purchase
- Storage

The raw CSV is intentionally excluded from Git. Download it and place it at:

```text
data/raw/all_tickets_processed_improved_v3.csv
```

The pipeline uses a **70/10/10/10 stratified split** (train/tune/calibration/test). The tune set is used for threshold selection. The calibration set is a separate held-out partition used solely to measure post-hoc calibration quality (ECE and Brier score) after the model is fitted — it is not used during training. `CalibratedClassifierCV` applies internal 5-fold cross-validation on the training set itself, independent of this split.

The category distribution is imbalanced, which is why model comparison includes **Macro F1** in addition to overall accuracy.

![Ticket category distribution](reports/figures/class_distribution.png)

## How it works

```mermaid
flowchart LR
    A[Ticket text] --> B[TF-IDF<br/>unigrams + bigrams]
    B --> C[Logistic Regression]
    C --> D[Class scores]
    D --> E{Confidence >= 0.54?}
    E -->|Yes| F[Auto-route]
    E -->|No| G[Human review]
```

The experimental TextCNN uses learned embeddings, parallel 1D convolutions with kernel sizes 3/4/5, global max pooling, dropout, and a linear classifier.

## System details

**Data handling:** The script checks for identical tickets that have conflicting labels and drops them before training. It also saves the exact train/tune/calibration/test IDs alongside a SHA-256 hash of the source dataset. If you modify the CSV later, the pipeline will refuse to load the stale splits, preventing accidental train/test leakage.

**Model integrity:** It's easy to deploy a new model but forget to update its threshold config. To prevent this, the JSON config stores the SHA-256 hash of the model that generated it. The API raises an exception at startup if the hashes don't match. 

**Observability:** The FastAPI app exposes standard Prometheus metrics at `/metrics` so you can monitor latency, request volume, confidence distributions, and how often tickets are being routed vs. flagged for review.

**Security:** The `/predict` endpoint checks for an `X-API-Key` header (using constant-time comparison). PyTorch weights are loaded with `weights_only=True` to avoid pickle exploits.

**CI/CD:** The repository uses `ruff` for formatting/linting and `mypy` in strict mode. A GitHub Actions workflow verifies these checks and runs the `pytest` suite on every push.

## Repository structure

```text
.
├── app/
│   └── main.py                     # FastAPI application
├── artifacts/
│   ├── baseline.joblib             # Local trained baseline model
│   └── cnn/                        # CNN config, vocabulary, labels, weights
├── data/
│   └── raw/                        # Dataset location (CSV ignored by Git)
├── notebook/                       # Exploratory notebooks
├── reports/
│   ├── data/                       # Persisted split manifests
│   ├── figures/                    # Generated plots
│   └── metrics/                    # Metrics, thresholds, predictions
├── src/
│   ├── data.py                     # Loading, deduplication, stratified splitting
│   ├── evaluate.py                 # Shared classification & calibration metrics
│   ├── routing_utils.py            # Threshold selection via exact Clopper-Pearson bounds + Bonferroni correction
│   ├── train_baseline.py           # TF-IDF + Logistic Regression training
│   ├── evaluate_tune_baseline.py   # Baseline tune-set predictions + calibration metrics (used for threshold selection)
│   ├── analyze_threshold_baseline.py
│   ├── evaluate_calibration_baseline.py  # Baseline calibration quality on held-out calibration partition
│   ├── evaluate_routing_baseline.py
│   ├── evaluate_baseline.py        # Baseline test evaluation
│   ├── cnn_data.py                 # CNN tokenization/vocabulary/dataset
│   ├── textcnn.py                  # TextCNN architecture
│   ├── train_cnn.py                # TextCNN training + early stopping
│   ├── evaluate_cnn.py             # TextCNN test evaluation
│   ├── evaluate_tune_cnn.py        # TextCNN tune-set predictions + calibration metrics
│   ├── analyze_threshold_cnn.py    # CNN threshold analysis
│   ├── evaluate_calibration_cnn.py # CNN calibration quality on held-out calibration partition
│   ├── compare_models.py           # Model comparison table
│   ├── select_model.py             # Automated production model selection
│   ├── error_summary.py            # CNN error analysis
│   ├── plot_results.py             # README/report figures
│   └── inference.py                # Baseline/CNN prediction backends
├── tests/
│   ├── test_api.py                 # FastAPI contract tests (endpoints, response shape, routing logic)
│   ├── test_api_hardening.py       # Security/hardening tests (auth modes, Prometheus label safety, input limits)
│   ├── test_ml_smoke.py            # Baseline predictor smoke test (train, serialize, load, and predict end-to-end)
│   ├── test_validation_design.py   # Data-split integrity and threshold selection (Clopper-Pearson bounds, disjointness, determinism)
│   └── test_integration.py         # ML pipeline integration tests
├── .github/
│   └── workflows/ci.yml            # GitHub Actions CI
├── .gitignore
├── pyproject.toml                  # Ruff / pytest configuration
├── requirements.txt                # CPU dependencies
└── requirements-cuda.txt           # GPU/CUDA override
```

Model binaries (`*.joblib`, `*.pt`) and the raw dataset are ignored by Git. A fresh clone requires placing the dataset in the correct location and using DVC to reproduce the pipeline before starting the API.

## Getting started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd it-service-ticket-classification
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The codebase requires NumPy, pandas, scikit-learn, joblib, Matplotlib, PyTorch, FastAPI, Uvicorn, and pytest.

| File | Purpose |
|---|---|
| `requirements-dev.txt` | Development + testing: all runtime deps plus pytest, ruff, mypy, and Prometheus client |
| `requirements-cuda.txt` | GPU/CUDA PyTorch override — replaces the CPU torch wheel for training or CNN serving on a GPU host |
| `requirements-train.txt` | Training-time extras: CNN runtime deps (filelock, fsspec, etc.) plus Matplotlib for evaluation plots |
| `requirements-cnn.txt` | CNN inference runtime: PyTorch dependency shims (filelock, fsspec, networkx, sympy, jinja2) |
| `requirements-mlops.txt` | MLOps tooling: DVC (with S3 remote support), MLflow, PyYAML, and SciPy |

```bash
pip install -r requirements-dev.txt
```

For GPU/CUDA environments (only needed for training or CNN serving), install with the CUDA override instead:

```bash
pip install -r requirements-cuda.txt
```

### 4. Add the dataset

Download the Kaggle dataset and place the CSV at:

```text
data/raw/all_tickets_processed_improved_v3.csv
```

### 5. Reproduce the pipeline (DVC)

This repository uses DVC for reproducibility. Once the raw dataset is in place, you can run the entire evaluation pipeline (including training and evaluation for both models) with a single command:

```bash
dvc repro
```

DVC will automatically skip any stages that are already up to date. To view the generated metrics:

```bash
dvc metrics show
```

### 6. Experiment Tracking (MLflow)

All training hyperparameters, metrics, and models are automatically tracked using MLflow. The tracking database is stored locally in `mlruns.db` (ignored by Git).

To view the experiments, run:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser.

To disable tracking, set the environment variable:
`MLFLOW_TRACKING_ENABLED=false`

## Run the API

The API uses the Logistic Regression backend by default.

```bash
python -m uvicorn app.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Example request:

```json
{
  "text": "I cannot access my account and need my password reset"
}
```

Example response shape:

```json
{
  "category": "Access",
  "confidence": 0.99,
  "threshold": 0.54,
  "needs_manual_review": false,
  "top_3": [
    {"category": "Access", "probability": 0.99},
    {"category": "Hardware", "probability": 0.01},
    {"category": "Storage", "probability": 0.0}
  ]
}
```

The API validates ticket length, returns the three highest-scoring categories, and flags predictions whose confidence is below the selected threshold.

## Run with Docker

A Dockerfile is provided to run the FastAPI application in an isolated container. It uses `python:3.12.14-slim` and installs only the runtime dependencies (excluding PyTorch for the baseline, and excluding DVC/MLflow entirely).

1. Build the baseline image:

```bash
docker build --pull -t it-ticket-baseline:test --build-arg MODEL_BACKEND=baseline .
```

2. Run the baseline container:

```bash
docker run --rm -p 8000:8000 -e MODEL_BACKEND=baseline -e APP_ENV=production -e API_KEY=test_key it-ticket-baseline:test
```

*(Note: The model artifacts must be generated by DVC before starting the container, as they are copied into the image during the build.)*

To build and use the TextCNN backend instead, ensure the CNN artifact exists, then rebuild and run with:
```bash
docker build --pull -t it-ticket-cnn:test --build-arg MODEL_BACKEND=cnn .
docker run --rm -p 8000:8000 -e MODEL_BACKEND=cnn -e APP_ENV=production -e API_KEY=test_key it-ticket-cnn:test
```

PowerShell example:
```powershell
docker run --rm -p 8000:8000 -e MODEL_BACKEND="baseline" -e APP_ENV=production -e API_KEY=test_key it-ticket-baseline:test
```

## Optional: running scripts directly

You can also run specific scripts directly without DVC, which can be useful during development. Note that DVC is the recommended way to reproduce experiments.

Train the neural model:

```bash
python -m src.train_cnn
```

The CNN can be served locally by setting `MODEL_BACKEND=cnn` before starting Uvicorn.

PowerShell:

```powershell
$env:MODEL_BACKEND="cnn"
python -m uvicorn app.main:app --reload
```

macOS/Linux:

```bash
MODEL_BACKEND=cnn python -m uvicorn app.main:app --reload
```

## Generate reports

Create the error summary after CNN evaluation:

```bash
python -m src.error_summary
```

Regenerate plots after the required metrics files exist:

```bash
python -m src.plot_results
```

Generated outputs are kept under [`reports/metrics/`](reports/metrics/) and [`reports/figures/`](reports/figures/).

## Tests

Run the full test suite from the repository root:

```bash
python -m pytest -v tests/test_api.py tests/test_api_hardening.py tests/test_ml_smoke.py tests/test_validation_design.py tests/test_integration.py tests/test_data_split.py
```

The tests are grouped by concern:

**API contract** (`test_api.py`)
- Root, health, and predict endpoint responses
- `needs_manual_review` routing flag matches confidence vs threshold
- Top-3 categories returned in descending probability order
- Empty-text rejection (HTTP 422)

**Security / hardening** (`test_api_hardening.py`)
- Auth modes: open (no key required) and keyed (401 on missing/wrong key)
- Prometheus label safety: arbitrary URLs use `UNMATCHED` sentinel, preventing unbounded cardinality
- Oversized and blank inputs rejected (HTTP 422)
- `/metrics` endpoint reachable

**ML smoke test** (`test_ml_smoke.py`)
- Trains a real TF-IDF + Logistic Regression pipeline on synthetic data, serializes it, loads it through `BaselinePredictor`, and asserts correct output structure, confidence bounds, and artifact SHA-256 handling

**Data-split integrity** (`test_data_split.py`)
- SHA-256 row IDs, split disjointness, blank-row rejection, manifest consistency

**Validation / data-split integrity** (`test_validation_design.py`)
- Clopper-Pearson and simultaneous lower-bound calculations
- Threshold selection determinism and coverage-maximizing choice
- Train/tune/test split disjointness
- Blank-row rejection and SHA-256 row-ID uniqueness
- Per-class Wilson lower-bound non-negativity

**Integration** (`test_integration.py`)
- End-to-end ML pipeline checks requiring DVC-pulled artifacts

## Design decisions

**Why Logistic Regression?**  
I built the TextCNN expecting it to capture semantics better, but TF-IDF + Logistic Regression actually outperformed it slightly across the board (Accuracy, F1, and routing coverage). It's also much faster to serve and requires fewer dependencies, so it's the practical choice for production here.

**Why evaluate with Macro F1?**  
The dataset is highly imbalanced. Using Macro F1 prevents the model from looking artificially good just by guessing the majority classes correctly.

## Limitations

- Although the production baseline is sigmoid-calibrated, calibration quality may change under distribution shift and should be monitored using production outcomes or human-review feedback.
- Several ticket categories have overlapping language, which creates unavoidable ambiguity for a text-only classifier.
- The current CNN tokenizer is whitespace-based and intentionally simple.
- Evaluation is based on one public dataset; production data may have different vocabulary and class distributions.

## Getting help

- Use the FastAPI Swagger UI at `/docs` for request/response examples.
- Check [`reports/metrics/`](reports/metrics/) for saved evaluation results and [`reports/figures/`](reports/figures/) for visualizations.
- For defects or feature requests, open an issue in the GitHub repository and include the command you ran, the backend (`baseline` or `cnn`), and the relevant error/output.

## Contributing

Contributions are welcome. Keep changes focused, avoid committing the raw dataset or generated model binaries, and run the test suite before opening a pull request:

```bash
python -m pytest -v
```

For larger changes, open an issue first so the proposed behavior can be discussed before implementation.

## Maintainer

Maintained by the repository owner. If the project is published under an organization or transferred to another maintainer, update this section with the appropriate contact or team information.

