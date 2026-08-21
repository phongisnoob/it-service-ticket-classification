import datetime
import json
import platform
import subprocess

import joblib
import sklearn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import load_data, load_splits
from src.evaluate import calculate_metrics
from src.hashing import calculate_file_sha256
from src.paths import ARTIFACT_DIR, DATA_PATH, METRICS_DIR, ROOT_DIR
from src.tracking import log_artifact, log_dict_as_artifact, log_metrics, log_params, start_run


def train_baseline() -> tuple[CalibratedClassifierCV, dict[str, float]]:
    df = load_data()
    splits = load_splits(df)
    train_df = splits.train
    tune_df = splits.tune

    X_train = train_df["Document"]
    y_train = train_df["Topic_group"]
    X_tune = tune_df["Document"]
    y_tune = tune_df["Topic_group"]

    with open(ROOT_DIR / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["baseline"]

    base_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(params["ngram_range"]),
                    min_df=params["min_df"],
                    max_df=params["max_df"],
                    sublinear_tf=True,
                    max_features=params["max_features"],
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=params["max_iter"], class_weight=params["class_weight"]),
            ),
        ]
    )

    print("Training and calibrating baseline with 5-fold CV...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline, method="sigmoid", cv=5, n_jobs=1
    )
    calibrated_model.fit(X_train, y_train)

    tune_predictions = calibrated_model.predict(X_tune)
    val_metrics = calculate_metrics(y_tune, tune_predictions)

    print("\nTune metrics")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")

    model_path = ARTIFACT_DIR / "baseline.joblib"
    joblib.dump(calibrated_model, model_path)

    model_sha256 = calculate_file_sha256(model_path)
    dataset_sha256 = calculate_file_sha256(DATA_PATH)

    try:
        git_commit: str | None = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None

    hyperparameters = {
        "ngram_range": params["ngram_range"],
        "min_df": params["min_df"],
        "max_df": params["max_df"],
        "max_features": params["max_features"],
        "max_iter": params["max_iter"],
        "class_weight": params["class_weight"],
        "calibration_method": "sigmoid (5-fold CV on training set)",
    }

    metadata: dict[str, object] = {
        "model_backend": "baseline",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "random_seed": 42,
        "hyperparameters": hyperparameters,
        "val_metrics": val_metrics,
    }

    with open(ARTIFACT_DIR / "baseline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(METRICS_DIR / "baseline_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)

    with start_run(run_name="train_baseline", model_backend="baseline"):
        log_params(hyperparameters)
        log_metrics(val_metrics)
        log_artifact(str(model_path), "artifacts")
        log_dict_as_artifact(metadata, "baseline_metadata.json")

    print(f"\nModel saved to {model_path} (SHA-256: {model_sha256[:12]}...)")

    return calibrated_model, val_metrics


def main() -> None:
    train_baseline()


if __name__ == "__main__":
    main()
