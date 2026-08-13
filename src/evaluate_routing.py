import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)


TEST_PREDICTIONS_PATH = (
    METRICS_DIR
    / "cnn_test_predictions.csv"
)

THRESHOLD_PATH = (
    METRICS_DIR
    / "selected_threshold.json"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "routing_metrics.json"
)


# ============================================================
# Load test predictions
# ============================================================

results = pd.read_csv(
    TEST_PREDICTIONS_PATH
)


# ============================================================
# Load threshold selected using VALIDATION set
# ============================================================

with open(
    THRESHOLD_PATH,
    "r",
) as f:

    threshold_config = json.load(f)


threshold = threshold_config[
    "threshold"
]


# ============================================================
# Apply threshold to TEST set
# ============================================================

auto_routed_mask = (
    results["confidence"]
    >= threshold
)

auto_routed = results[
    auto_routed_mask
]

manual_review = results[
    ~auto_routed_mask
]


# ============================================================
# Metrics
# ============================================================

total_tickets = len(results)

auto_routed_count = len(
    auto_routed
)

manual_review_count = len(
    manual_review
)


coverage = (
    auto_routed_count
    / total_tickets
)

manual_review_rate = (
    manual_review_count
    / total_tickets
)


if auto_routed_count > 0:

    selective_accuracy = (
        auto_routed["correct"]
        .mean()
    )

else:

    selective_accuracy = 0.0


overall_accuracy = (
    results["correct"]
    .mean()
)


# ============================================================
# Print results
# ============================================================

print("\nFinal Routing Evaluation")
print("=" * 50)

print(
    f"Selected threshold:       "
    f"{threshold:.2f}"
)

print(
    f"Overall test accuracy:    "
    f"{overall_accuracy:.2%}"
)

print(
    f"Auto-route coverage:      "
    f"{coverage:.2%}"
)

print(
    f"Manual review rate:       "
    f"{manual_review_rate:.2%}"
)

print(
    f"Auto-routed accuracy:     "
    f"{selective_accuracy:.2%}"
)

print(
    f"\nTotal tickets:            "
    f"{total_tickets}"
)

print(
    f"Auto-routed tickets:      "
    f"{auto_routed_count}"
)

print(
    f"Manual review tickets:    "
    f"{manual_review_count}"
)


# ============================================================
# Save metrics
# ============================================================

routing_metrics = {
    "threshold":
        threshold,

    "overall_test_accuracy":
        float(overall_accuracy),

    "coverage":
        float(coverage),

    "manual_review_rate":
        float(manual_review_rate),

    "auto_routed_accuracy":
        float(selective_accuracy),

    "total_tickets":
        int(total_tickets),

    "auto_routed_tickets":
        int(auto_routed_count),

    "manual_review_tickets":
        int(manual_review_count),
}


with open(
    OUTPUT_PATH,
    "w",
) as f:

    json.dump(
        routing_metrics,
        f,
        indent=4,
    )


print(
    f"\nSaved to: {OUTPUT_PATH}"
)