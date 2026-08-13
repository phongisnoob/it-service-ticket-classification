from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, split_data
from src.evaluate import calculate_calibration_metrics

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    df = load_data()
    _, val_df, _ = split_data(df, random_state=42)

    probabilities = model.predict_proba(val_df["Document"])
    predictions = model.predict(val_df["Document"])
    confidence = np.max(probabilities, axis=1)

    results = pd.DataFrame({
        "ticket_id": val_df.index.to_numpy(),
        "true_label": val_df["Topic_group"].values,
        "predicted_label": predictions,
        "confidence": confidence,
    })

    results["correct"] = results["true_label"] == results["predicted_label"]
    output_path = METRICS_DIR / "baseline_val_predictions.csv"
    results.to_csv(output_path, index=False)

    calibration = calculate_calibration_metrics(
        results["true_label"],
        results["predicted_label"],
        results["confidence"],
    )

    calibration_path = METRICS_DIR / "baseline_calibration_metrics.json"
    import json
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=4)

    print("Validation accuracy:", results["correct"].mean())
    print("Brier score:", calibration["brier_score"])
    print("ECE:", calibration["expected_calibration_error"])
    print("Saved to:", output_path)
    print("Saved calibration metrics to:", calibration_path)


if __name__ == "__main__":
    main()