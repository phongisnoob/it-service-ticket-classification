"""Evaluate the baseline model on the tune set.

Produces:
- Tune-set predictions CSV (used by threshold selection)
- Tune-set calibration metrics (ECE, Brier score)

Note: these calibration metrics are measured on the TUNE set — the same
partition used for threshold selection.  For held-out calibration quality
assessment on the independent CALIBRATION partition, see
``evaluate_calibration_baseline.py``.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, load_splits
from src.evaluate import calculate_calibration_metrics
from src.paths import METRICS_DIR, ROOT_DIR

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = load_data()
    splits = load_splits(df)
    tune_df = splits.tune

    probabilities = model.predict_proba(tune_df["Document"])
    predictions = model.predict(tune_df["Document"])
    confidence = np.max(probabilities, axis=1)

    results = pd.DataFrame(
        {
            "ticket_id": tune_df.index.to_numpy(),
            "true_label": tune_df["Topic_group"].values,
            "predicted_label": predictions,
            "confidence": confidence,
        }
    )

    results["correct"] = results["true_label"] == results["predicted_label"]
    output_path = METRICS_DIR / "baseline_tune_predictions.csv"
    results.to_csv(output_path, index=False)

    calibration = calculate_calibration_metrics(
        results["true_label"],
        results["predicted_label"],
        results["confidence"],
    )

    calibration_path = METRICS_DIR / "baseline_tune_calibration_metrics.json"
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=4)

    print("Tune accuracy:", results["correct"].mean())
    print("Top label Brier score:", calibration["top_label_brier_score"])
    print("ECE:", calibration["expected_calibration_error"])
    print("Saved predictions to:", output_path)
    print("Saved tune calibration metrics to:", calibration_path)


if __name__ == "__main__":
    main()
