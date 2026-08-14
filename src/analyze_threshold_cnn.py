import json

import numpy as np
import pandas as pd

from src.hashing import calculate_file_sha256
from src.paths import METRICS_DIR, ROOT_DIR
from src.routing_utils import compute_bootstrap_ci

INPUT_PATH = METRICS_DIR / "cnn_val_predictions.csv"
OUTPUT_PATH = METRICS_DIR / "threshold_analysis.csv"
THRESHOLD_PATH = METRICS_DIR / "selected_threshold.json"
MODEL_PATH = ROOT_DIR / "artifacts" / "cnn" / "textcnn.pt"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input predictions not found: {INPUT_PATH}")

    results = pd.read_csv(INPUT_PATH)
    thresholds = np.round(np.arange(0.10, 1.00, 0.01), 2)
    rows = []

    for threshold in thresholds:
        routed = results[results["confidence"] >= threshold]
        coverage = len(routed) / len(results)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": float(coverage),
                "auto_routed_accuracy": float(routed["correct"].mean()) if len(routed) > 0 else 0.0,
                "manual_review_rate": float(1.0 - coverage),
            }
        )

    threshold_results = pd.DataFrame(rows)

    print("\nThreshold Analysis (CNN)")
    print("=" * 75)
    print(threshold_results.round(4).to_string(index=False))

    threshold_results.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved threshold analysis to:", OUTPUT_PATH)

    TARGET_ACCURACY = 0.90
    candidates = threshold_results[
        threshold_results["auto_routed_accuracy"] >= TARGET_ACCURACY
    ].sort_values("coverage", ascending=False)

    selected_row = None
    selected_acc_ci: list[float] = []
    selected_cov_ci: list[float] = []

    for _, candidate in candidates.iterrows():
        acc_ci, cov_ci = compute_bootstrap_ci(results, float(candidate["threshold"]))
        if acc_ci[0] >= TARGET_ACCURACY:
            selected_row = candidate
            selected_acc_ci = acc_ci
            selected_cov_ci = cov_ci
            break

    if selected_row is not None:
        selected_threshold = float(selected_row["threshold"])

        print("\nSelected Threshold")
        print("=" * 75)
        print(f"Threshold:            {selected_threshold:.2f}")
        print(
            f"Coverage:             {selected_row['coverage']:.2%} "
            f"(95% CI: [{selected_cov_ci[0]:.2%}, {selected_cov_ci[1]:.2%}])"
        )
        print(
            f"Auto-routed accuracy: {selected_row['auto_routed_accuracy']:.2%} "
            f"(95% CI: [{selected_acc_ci[0]:.2%}, {selected_acc_ci[1]:.2%}])"
        )
        print(f"Manual review rate:   {selected_row['manual_review_rate']:.2%}")

        threshold_config: dict[str, object] = {
            "threshold": selected_threshold,
            "target_accuracy": TARGET_ACCURACY,
            "selection_rule": (
                "maximize_coverage_subject_to_95pct_ci_lower_bound_gte_target_accuracy"
            ),
            "model_sha256": calculate_file_sha256(MODEL_PATH),
            "validation_coverage": float(selected_row["coverage"]),
            "validation_auto_routed_accuracy": float(selected_row["auto_routed_accuracy"]),
            "validation_accuracy_ci_lower": selected_acc_ci[0],
            "validation_manual_review_rate": float(selected_row["manual_review_rate"]),
            "bootstrap_ci_95": {
                "auto_routed_accuracy": selected_acc_ci,
                "coverage": selected_cov_ci,
            },
        }

        with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(threshold_config, f, indent=4)

        print("\nSaved selected threshold to:", THRESHOLD_PATH)

    else:
        best = threshold_results.loc[threshold_results["auto_routed_accuracy"].idxmax()]

        print(
            f"\nNo threshold satisfied the routing requirement:"
            f" lower 95% accuracy CI >= {TARGET_ACCURACY:.0%}."
        )
        print("\nBest validation result")
        print("=" * 75)
        print(f"Threshold:            {best['threshold']:.2f}")
        print(f"Coverage:             {best['coverage']:.2%}")
        print(f"Auto-routed accuracy: {best['auto_routed_accuracy']:.2%}")
        print(f"Manual review rate:   {best['manual_review_rate']:.2%}")

        THRESHOLD_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "No threshold has a 95% accuracy CI lower bound "
            f">= {TARGET_ACCURACY:.0%}. "
            "The previous threshold was removed."
        )


if __name__ == "__main__":
    main()
