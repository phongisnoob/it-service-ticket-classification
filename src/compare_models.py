import json

import pandas as pd

from src.paths import METRICS_DIR


def main() -> None:
    baseline_path = METRICS_DIR / "baseline_metrics.json"
    baseline_calib_path = METRICS_DIR / "baseline_tune_calibration_metrics.json"
    cnn_path = METRICS_DIR / "cnn_metrics.json"

    if not baseline_path.exists() or not cnn_path.exists():
        print("Metrics files not found for comparison.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    if baseline_calib_path.exists():
        with open(baseline_calib_path, "r", encoding="utf-8") as f:
            baseline_calib = json.load(f)
            baseline.update(baseline_calib)

    with open(cnn_path, "r", encoding="utf-8") as f:
        cnn = json.load(f)

    comparison = pd.DataFrame(
        [
            {
                "model": "TF-IDF + Logistic Regression",
                **baseline,
            },
            {
                "model": "TextCNN",
                **cnn,
            },
        ]
    )

    print(comparison.to_string(index=False))

    output_path = METRICS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
