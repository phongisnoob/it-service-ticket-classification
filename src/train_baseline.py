from pathlib import Path
import json

import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_data, split_data
from src.evaluate import calculate_metrics, print_report


ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():

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
    # Model
    # -----------------------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=100_000,
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )
        ),
    ])


    # -----------------------------------------
    # Train
    # -----------------------------------------

    print("Training baseline...")

    model.fit(
        X_train,
        y_train,
    )


    # -----------------------------------------
    # Validation
    # -----------------------------------------

    val_predictions = model.predict(X_val)

    val_metrics = calculate_metrics(
        y_val,
        val_predictions,
    )

    print("\nValidation metrics")

    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")


    # -----------------------------------------
    # Save model
    # -----------------------------------------

    model_path = (
        ARTIFACT_DIR
        / "baseline.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved to {model_path}"
    )


    # -----------------------------------------
    # Save metrics
    # -----------------------------------------

    with open(
        METRICS_DIR / "baseline_val_metrics.json",
        "w",
    ) as f:

        json.dump(
            val_metrics,
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()