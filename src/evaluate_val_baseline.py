from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, split_data


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "artifacts"
    / "baseline.joblib"
)

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Load model
model = joblib.load(
    MODEL_PATH
)


# Recreate split
df = load_data()

_, val_df, _ = split_data(
    df,
    random_state=42,
)


# Predictions
probabilities = model.predict_proba(
    val_df["Document"]
)

predictions = model.predict(
    val_df["Document"]
)


confidence = np.max(
    probabilities,
    axis=1,
)


# Save validation predictions
results = pd.DataFrame({
    "true_label":
        val_df["Topic_group"].values,

    "predicted_label":
        predictions,

    "confidence":
        confidence,
})


results["correct"] = (
    results["true_label"]
    ==
    results["predicted_label"]
)


output_path = (
    METRICS_DIR
    / "baseline_val_predictions.csv"
)


results.to_csv(
    output_path,
    index=False,
)


print(
    "Validation accuracy:",
    results["correct"].mean()
)

print(
    "Saved to:",
    output_path
)