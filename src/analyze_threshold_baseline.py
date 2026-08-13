import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.routing_utils import compute_bootstrap_ci


ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT_DIR / "reports" / "metrics"
INPUT_PATH = METRICS_DIR / "baseline_val_predictions.csv"
OUTPUT_PATH = METRICS_DIR / "baseline_threshold_analysis.csv"
THRESHOLD_PATH = METRICS_DIR / "baseline_selected_threshold.json"
MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"


def calculate_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input predictions not found: {INPUT_PATH}")

    results = pd.read_csv(INPUT_PATH)

    # Fine threshold grid: 0.10 to 0.99 with 0.01 step
    thresholds = np.round(np.arange(0.10, 1.00, 0.01), 2)
    rows = []

    for threshold in thresholds:
        auto_routed_mask = results["confidence"] >= threshold
        routed = results[auto_routed_mask]
        coverage = len(routed) / len(results)
        routed_accuracy = routed["correct"].mean() if len(routed) > 0 else 0.0
        manual_review_rate = 1.0 - coverage

        rows.append({
            "threshold": round(float(threshold), 2),
            "coverage": float(coverage),
            "auto_routed_accuracy": float(routed_accuracy),
            "manual_review_rate": float(manual_review_rate),
        })

    threshold_results = pd.DataFrame(rows)

    print("\nThreshold Analysis (Baseline)")
    print("=" * 75)
    print(threshold_results.round(4).to_string(index=False))

    threshold_results.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved threshold analysis to:", OUTPUT_PATH)

    TARGET_ACCURACY = 0.90
    candidates = threshold_results[
        threshold_results["auto_routed_accuracy"] >= TARGET_ACCURACY
    ]

    if len(candidates) > 0:
        selected = candidates.sort_values("coverage", ascending=False).iloc[0]
        selected_threshold = float(selected["threshold"])

        acc_ci, cov_ci = compute_bootstrap_ci(results, selected_threshold)

        print("\nSelected Threshold")
        print("=" * 75)
        print(f"Threshold:            {selected_threshold:.2f}")
        print(f"Coverage:             {selected['coverage']:.2%} (95% CI: [{cov_ci[0]:.2%}, {cov_ci[1]:.2%}])")
        print(f"Auto-routed accuracy: {selected['auto_routed_accuracy']:.2%} (95% CI: [{acc_ci[0]:.2%}, {acc_ci[1]:.2%}])")
        print(f"Manual review rate:   {selected['manual_review_rate']:.2%}")

        threshold_config = {
            "threshold": selected_threshold,
            "target_accuracy": TARGET_ACCURACY,
            "model_sha256": calculate_sha256(MODEL_PATH),
            "validation_coverage": float(selected["coverage"]),
            "validation_auto_routed_accuracy": float(selected["auto_routed_accuracy"]),
            "validation_manual_review_rate": float(selected["manual_review_rate"]),
            "bootstrap_ci_95": {
                "auto_routed_accuracy": acc_ci,
                "coverage": cov_ci,
            },
        }

        with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(threshold_config, f, indent=4)

        print("\nSaved selected threshold to:", THRESHOLD_PATH)

    else:
        best = threshold_results.loc[
            threshold_results["auto_routed_accuracy"].idxmax()
        ]
        print(f"\nNo threshold achieved {TARGET_ACCURACY:.0%} auto-routed accuracy.")
        print("\nBest validation result")
        print("=" * 75)
        print(f"Threshold:            {best['threshold']:.2f}")
        print(f"Coverage:             {best['coverage']:.2%}")
        print(f"Auto-routed accuracy: {best['auto_routed_accuracy']:.2%}")
        print(f"Manual review rate:   {best['manual_review_rate']:.2%}")

        THRESHOLD_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "No threshold meets the target accuracy. The previous threshold was removed."
        )


if __name__ == "__main__":
    main()