"""Final routing evaluation for the baseline model on the held-out test set.

Methodology note
----------------
The routing threshold was selected on the **tune set** using simultaneous
one-sided exact Clopper-Pearson confidence bounds with a Bonferroni correction
(see src/analyze_threshold_baseline.py and params.yaml[routing]).

This script evaluates the *pre-selected* threshold on the **test set** and
reports point estimates only.  No confidence interval is re-computed here
because:

1. The threshold was chosen before any test-set contact — applying a CI
   procedure retrospectively on the same data used for reporting would be
   post-hoc and misleading.
2. The statistical guarantee (lower bound >= target accuracy) was established
   on the tune set.  Test-set results are descriptive evidence, not a new
   guarantee.

If a CI is desired for publication or monitoring, use a fresh bootstrap on a
*new* batch of production data that was never part of threshold selection.
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, split_data
from src.paths import BASELINE_MODEL_PATH as MODEL_PATH
from src.paths import METRICS_DIR
from src.tracking import log_metrics, start_run

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
    selection_rule = threshold_config.get("selection_rule", "unknown")
    target_accuracy = float(threshold_config.get("target_accuracy", 0.90))

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
    overall_accuracy = float(results["correct"].mean())
    coverage = len(auto_routed) / total_tickets
    manual_review_rate = len(manual_review) / total_tickets
    auto_routed_accuracy = float(
        auto_routed["correct"].mean() if len(auto_routed) > 0 else 0.0
    )

    print("\nBaseline Final Routing Evaluation (Test Set)")
    print("=" * 55)
    print(f"Threshold (selected on tune set): {threshold:.2f}")
    print(f"Selection rule:                   {selection_rule}")
    print(f"Target accuracy (tune set, CI):   >= {target_accuracy:.0%}")
    print(f"Overall test accuracy:            {overall_accuracy:.2%}")
    print(f"Auto-route coverage:              {coverage:.2%}")
    print(f"Manual review rate:               {manual_review_rate:.2%}")
    print(f"Auto-routed accuracy (point):     {auto_routed_accuracy:.2%}")
    print(f"\nTotal tickets:                    {total_tickets}")
    print(f"Auto-routed tickets:              {len(auto_routed)}")
    print(f"Manual review tickets:            {len(manual_review)}")

    metrics: dict[str, object] = {
        "threshold": threshold,
        "threshold_selection_rule": selection_rule,
        "target_accuracy_on_tune_set": target_accuracy,
        "overall_test_accuracy": overall_accuracy,
        "coverage": float(coverage),
        "manual_review_rate": float(manual_review_rate),
        "auto_routed_accuracy": auto_routed_accuracy,
        "total_tickets": int(total_tickets),
        "auto_routed_tickets": int(len(auto_routed)),
        "manual_review_tickets": int(len(manual_review)),
        "note": (
            "Test-set metrics are point estimates. "
            "The statistical accuracy guarantee was established on the tune set "
            "using simultaneous Clopper-Pearson bounds; see baseline_selected_threshold.json."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with start_run(run_name="evaluate_routing_baseline", model_backend="baseline"):
        log_metrics({
            "test_routing_threshold": threshold,
            "test_overall_accuracy": overall_accuracy,
            "test_auto_route_coverage": float(coverage),
            "test_manual_review_rate": float(manual_review_rate),
            "test_auto_routed_accuracy": auto_routed_accuracy,
        })

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
