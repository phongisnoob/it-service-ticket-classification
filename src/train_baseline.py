from typing import Any
import datetime
import json
import platform
import subprocess
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import DATA_PATH, calculate_file_sha256, load_data, split_data
from src.evaluate import calculate_metrics

ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline() -> Any:
    # -----------------------------------------
    # Load and split data
    # -----------------------------------------
    df = load_data()
    train_df, val_df, _ = split_data(df)

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]

    X_val = val_df["Document"]
    y_val = val_df["Topic_group"]

    # -----------------------------------------
    # Base Model Pipeline
    # -----------------------------------------
    base_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # -----------------------------------------
    # Calibration Stage
    # -----------------------------------------
    print("Training and calibrating baseline with 5-fold CV...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    calibrated_model.fit(X_train, y_train)

    # -----------------------------------------
    # Validation
    # -----------------------------------------
    val_predictions = calibrated_model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_predictions)

    print("\nValidation metrics")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # -----------------------------------------
    # Save model & metadata
    # -----------------------------------------
    model_path = ARTIFACT_DIR / "baseline.joblib"
    joblib.dump(calibrated_model, model_path)

    model_sha256 = calculate_file_sha256(model_path)
    dataset_sha256 = calculate_file_sha256(DATA_PATH)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None

    metadata = {
        "model_backend": "baseline",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "random_seed": 42,
        "hyperparameters": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 100000,
            "max_iter": 2000,
            "class_weight": "balanced",
            "calibration_method": "sigmoid (5-fold CV on training set)",
        },
        "val_metrics": val_metrics,
    }

    with open(
        ARTIFACT_DIR / "baseline_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)

    print(f"\nModel saved to {model_path} (SHA-256: {model_sha256[:12]}...)")

    # -----------------------------------------
    # Save metrics
    # -----------------------------------------
    with open(METRICS_DIR / "baseline_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)

    return calibrated_model, val_metrics


def main() -> None:
    train_baseline()


if __name__ == "__main__":
    main()
