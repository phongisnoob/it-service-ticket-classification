
# IT Service Ticket Classification & Confidence-Based Routing

An NLP project that sorts IT service desk tickets into eight support categories. Instead of forcing a prediction on every ticket, the system flags ambiguous cases for manual review based on model confidence.

I started this project to compare a classical TF-IDF + Logistic Regression baseline against a PyTorch TextCNN. The Logistic Regression model ended up performing slightly better on the test set, so that's what runs in the FastAPI backend by default.

## Overview

Most classification tutorials just try to maximize accuracy. In a real helpdesk, you can't afford to auto-route ambiguous tickets. This project focuses on the practical trade-off between coverage (how many tickets we automate) and accuracy (how often the automated routing is correct).

Key details:
- Trained on a public dataset of **47,837 tickets**, stratified into a 70/15/15 split.
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

I initially tried picking the routing threshold by just checking accuracy on the validation split. However, to avoid picking a "lucky" threshold that overfits, the script now computes a 95% bootstrap confidence interval (1,000 resamples). It requires the **lower bound** of the accuracy CI to be at least 90%, then picks the threshold that maximizes coverage.

For Logistic Regression, the chosen threshold was **0.54**.

| Test-set routing metric | Result |
|---|---:|
| Overall accuracy | 85.31% |
| Auto-route coverage | **86.62%** |
| Accuracy on auto-routed tickets | **90.57%** |
| Manual-review rate | 13.38% |
| Auto-routed tickets | 6,216 / 7,176 |

![Routing accuracy versus coverage](reports/figures/baseline_threshold_tradeoff.png)

On the held-out test set, auto-routed tickets achieved 90.57% accuracy. The 95% bootstrap confidence interval was approximately 89.84%–91.27%, so the 90% target should be interpreted as a validation-time routing criterion rather than a guaranteed lower bound on future data.

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

**Data handling:** The script checks for identical tickets that have conflicting labels and drops them before training. It also saves the exact train/validation/test IDs alongside a SHA-256 hash of the source dataset. If you modify the CSV later, the pipeline will refuse to load the stale splits, preventing accidental train/test leakage.

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
│   ├── routing_utils.py            # Bootstrap CI for routing evaluation
│   ├── train_baseline.py           # TF-IDF + Logistic Regression training
│   ├── evaluate_val_baseline.py    # Baseline validation predictions
│   ├── analyze_threshold_baseline.py
│   ├── evaluate_routing_baseline.py
│   ├── evaluate_baseline.py        # Baseline test evaluation
│   ├── cnn_data.py                 # CNN tokenization/vocabulary/dataset
│   ├── textcnn.py                  # TextCNN architecture
│   ├── train_cnn.py                # TextCNN training + early stopping
│   ├── evaluate_cnn.py             # TextCNN test evaluation
│   ├── evaluate_val_cnn.py         # TextCNN validation predictions
│   ├── analyze_threshold_cnn.py    # CNN threshold analysis
│   ├── compare_models.py           # Model comparison table
│   ├── select_model.py             # Automated production model selection
│   ├── error_summary.py            # CNN error analysis
│   ├── plot_results.py             # README/report figures
│   └── inference.py                # Baseline/CNN prediction backends
├── tests/
│   ├── test_api.py                 # FastAPI unit tests
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
cd it-ticket-classification
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

```bash
pip install -r requirements.txt
```

For GPU/CUDA environments, install with the CUDA override instead:

```bash
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126
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

A Dockerfile is provided to run the FastAPI application in an isolated container. It installs only the runtime dependencies (excluding DVC and MLflow).

1. Build the image:

```bash
docker build -t it-ticket-api .
```

2. Run the container:

```bash
docker run --rm -p 8000:8000 -e MODEL_BACKEND=baseline it-ticket-api
```

*(Note: The model artifact must be generated by DVC before starting the container, as it is copied into the image during the build.)*

To use the TextCNN backend instead, ensure the CNN artifact exists, then rebuild and run with:
```bash
docker run --rm -p 8000:8000 -e MODEL_BACKEND=cnn it-ticket-api
```

PowerShell example:
```powershell
docker run --rm -p 8000:8000 -e MODEL_BACKEND="baseline" it-ticket-api
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

Run the API test suite from the repository root:

```bash
python -m pytest -v
```

The current tests cover:

- root and health endpoints
- valid prediction responses
- confidence/threshold routing logic
- descending top-3 scores
- rejection of empty ticket text

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

