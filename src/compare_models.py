import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)


def main():
    baseline_path = METRICS_DIR / "baseline_metrics.json"
    cnn_path = METRICS_DIR / "cnn_metrics.json"

    if not baseline_path.exists() or not cnn_path.exists():
        print("Metrics files not found for comparison.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    with open(cnn_path, "r", encoding="utf-8") as f:
        cnn = json.load(f)

    comparison = pd.DataFrame([
        {
            "model": "TF-IDF + Logistic Regression",
            **baseline,
        },
        {
            "model": "TextCNN",
            **cnn,
        },
    ])

    print(comparison.to_string(index=False))

    output_path = METRICS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()