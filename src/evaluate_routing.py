import json

import pandas as pd

from src.paths import METRICS_DIR

# ============================================================
# Paths
# ============================================================


TEST_PREDICTIONS_PATH = METRICS_DIR / "cnn_test_predictions.csv"

THRESHOLD_PATH = METRICS_DIR / "selected_threshold.json"

OUTPUT_PATH = METRICS_DIR / "routing_metrics.json"


def main():
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Test predictions not found: {TEST_PREDICTIONS_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold config not found: {THRESHOLD_PATH}")

    results = pd.read_csv(TEST_PREDICTIONS_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    threshold = float(threshold_config["threshold"])

    auto_routed_mask = results["confidence"] >= threshold
    auto_routed = results[auto_routed_mask]
    manual_review = results[~auto_routed_mask]

    total_tickets = len(results)
    auto_routed_count = len(auto_routed)
    manual_review_count = len(manual_review)

    coverage = auto_routed_count / total_tickets
    manual_review_rate = manual_review_count / total_tickets
    selective_accuracy = auto_routed["correct"].mean() if auto_routed_count > 0 else 0.0
    overall_accuracy = results["correct"].mean()

    print("\nFinal Routing Evaluation (CNN)")
    print("=" * 50)
    print(f"Selected threshold:       {threshold:.2f}")
    print(f"Overall test accuracy:    {overall_accuracy:.2%}")
    print(f"Auto-route coverage:      {coverage:.2%}")
    print(f"Manual review rate:       {manual_review_rate:.2%}")
    print(f"Auto-routed accuracy:     {selective_accuracy:.2%}")
    print(f"\nTotal tickets:            {total_tickets}")
    print(f"Auto-routed tickets:      {auto_routed_count}")
    print(f"Manual review tickets:    {manual_review_count}")

    routing_metrics = {
        "threshold": threshold,
        "overall_test_accuracy": float(overall_accuracy),
        "coverage": float(coverage),
        "manual_review_rate": float(manual_review_rate),
        "auto_routed_accuracy": float(selective_accuracy),
        "total_tickets": int(total_tickets),
        "auto_routed_tickets": int(auto_routed_count),
        "manual_review_tickets": int(manual_review_count),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(routing_metrics, f, indent=4)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
