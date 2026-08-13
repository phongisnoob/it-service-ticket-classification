import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, split_data
from src.routing_utils import compute_bootstrap_ci

# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"

METRICS_DIR = ROOT_DIR / "reports" / "metrics"

THRESHOLD_PATH = METRICS_DIR / "baseline_selected_threshold.json"

OUTPUT_PATH = METRICS_DIR / "baseline_routing_metrics.json"


def main() -> None:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold config not found: {THRESHOLD_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    threshold = float(threshold_config["threshold"])
    model = joblib.load(MODEL_PATH)
    df = load_data()
    _, _, test_df = split_data(df, random_state=42)

    X_test = test_df["Document"]
    y_test = test_df["Topic_group"].values

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    confidence = np.max(probabilities, axis=1)

    results = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_label": predictions,
            "confidence": confidence,
        }
    )
    results["correct"] = results["true_label"] == results["predicted_label"]

    auto_routed_mask = results["confidence"] >= threshold
    auto_routed = results[auto_routed_mask]
    manual_review = results[~auto_routed_mask]

    total_tickets = len(results)
    overall_accuracy = results["correct"].mean()
    coverage = len(auto_routed) / total_tickets
    manual_review_rate = len(manual_review) / total_tickets
    auto_routed_accuracy = auto_routed["correct"].mean() if len(auto_routed) > 0 else 0.0

    print("\nBaseline Final Routing Evaluation")
    print("=" * 55)
    print(f"Selected threshold:       {threshold:.2f}")
    print(f"Overall test accuracy:    {overall_accuracy:.2%}")
    print(f"Auto-route coverage:      {coverage:.2%}")
    print(f"Manual review rate:       {manual_review_rate:.2%}")
    print(f"Auto-routed accuracy:     {auto_routed_accuracy:.2%}")

    acc_ci, cov_ci = compute_bootstrap_ci(results, threshold)
    print(f"  95% CI accuracy:        [{acc_ci[0]:.2%}, {acc_ci[1]:.2%}]")
    print(f"  95% CI coverage:        [{cov_ci[0]:.2%}, {cov_ci[1]:.2%}]")

    print(f"\nTotal tickets:            {total_tickets}")
    print(f"Auto-routed tickets:      {len(auto_routed)}")
    print(f"Manual review tickets:    {len(manual_review)}")

    metrics = {
        "threshold": threshold,
        "overall_test_accuracy": float(overall_accuracy),
        "coverage": float(coverage),
        "manual_review_rate": float(manual_review_rate),
        "auto_routed_accuracy": float(auto_routed_accuracy),
        "bootstrap_ci_95": {
            "auto_routed_accuracy": acc_ci,
            "coverage": cov_ci,
        },
        "total_tickets": int(total_tickets),
        "auto_routed_tickets": int(len(auto_routed)),
        "manual_review_tickets": int(len(manual_review)),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
