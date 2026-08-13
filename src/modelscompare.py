import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)


with open(
    METRICS_DIR
    / "baseline_metrics.json"
) as f:

    baseline = json.load(f)


with open(
    METRICS_DIR
    / "cnn_metrics.json"
) as f:

    cnn = json.load(f)


comparison = pd.DataFrame([
    {
        "model":
            "TF-IDF + Logistic Regression",

        **baseline,
    },

    {
        "model":
            "TextCNN",

        **cnn,
    },
])


print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    METRICS_DIR
    / "model_comparison.csv",
    index=False,
)