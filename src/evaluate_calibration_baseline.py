"""Held-out calibration evaluation for the baseline model.

Evaluates ECE and Brier score on the CALIBRATION partition — the independent
10% hold-out that is never used for training or threshold selection.

Methodology note
----------------
Model probability calibration is performed via ``CalibratedClassifierCV``
with 5-fold cross-validation on the TRAINING set only (see
``train_baseline.py``).  This script measures the quality of that
calibration on unseen data by computing:

- Expected Calibration Error (ECE): mean absolute difference between
  confidence and correctness, aggregated across confidence bins.
- Top-label Brier score: mean squared error between max predicted
  probability and binary correctness indicator.

These figures describe how well the model's confidence scores represent
its actual accuracy on held-out data.  They are descriptive; no threshold
selection or model selection is performed here.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.data import load_calibration_split, load_data
from src.evaluate import calculate_calibration_metrics
from src.paths import METRICS_DIR, ROOT_DIR

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"
OUTPUT_PATH = METRICS_DIR / "baseline_calibration_holdout_metrics.json"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = load_data()
    calib_df = load_calibration_split(df)

    probabilities = model.predict_proba(calib_df["Document"])
    predictions = model.predict(calib_df["Document"])
    confidence = np.max(probabilities, axis=1)

    results = pd.DataFrame(
        {
            "ticket_id": calib_df.index.to_numpy(),
            "true_label": calib_df["Topic_group"].values,
            "predicted_label": predictions,
            "confidence": confidence,
        }
    )
    results["correct"] = results["true_label"] == results["predicted_label"]

    calibration = calculate_calibration_metrics(
        results["true_label"],
        results["predicted_label"],
        results["confidence"],
    )

    output: dict[str, object] = {
        "partition": "calibration_holdout",
        "n_samples": int(len(results)),
        "accuracy": float(results["correct"].mean()),
        **calibration,
        "note": (
            "Calibration metrics computed on the held-out CALIBRATION partition "
            "(10% of data). This partition was never used for training or "
            "threshold selection. The model's sigmoid calibration was fitted "
            "internally on the training set via 5-fold cross-validation."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Calibration holdout accuracy: {results['correct'].mean():.4f}")
    print(f"Top label Brier score: {calibration['top_label_brier_score']:.6f}")
    print(f"Expected Calibration Error (ECE): {calibration['expected_calibration_error']:.6f}")
    print(f"n_samples: {len(results)}")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
