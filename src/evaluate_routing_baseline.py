import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_data, split_data


# ============================================================
# Paths
# ============================================================

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

THRESHOLD_PATH = (
    METRICS_DIR
    / "baseline_selected_threshold.json"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "baseline_routing_metrics.json"
)


# ============================================================
# Load threshold selected using VALIDATION set
# ============================================================

with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8",
) as f:
    threshold_config = json.load(f)


threshold = float(
    threshold_config["threshold"]
)


# ============================================================
# Load trained baseline model
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# Recreate TEST split
# ============================================================

df = load_data()

_, _, test_df = split_data(
    df,
    random_state=42,
)


X_test = test_df["Document"]

y_test = test_df["Topic_group"].values


# ============================================================
# Predictions + confidence
# ============================================================

probabilities = model.predict_proba(
    X_test
)

predictions = model.predict(
    X_test
)

confidence = np.max(
    probabilities,
    axis=1,
)


# ============================================================
# Build results DataFrame
# ============================================================

results = pd.DataFrame({
    "true_label": y_test,
    "predicted_label": predictions,
    "confidence": confidence,
})


results["correct"] = (
    results["true_label"]
    ==
    results["predicted_label"]
)


# ============================================================
# Apply frozen threshold
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

overall_accuracy = (
    results["correct"].mean()
)

coverage = (
    len(auto_routed)
    / total_tickets
)

manual_review_rate = (
    len(manual_review)
    / total_tickets
)


if len(auto_routed) > 0:

    auto_routed_accuracy = (
        auto_routed["correct"].mean()
    )

else:

    auto_routed_accuracy = 0.0


# ============================================================
# Print
# ============================================================

print("\nBaseline Final Routing Evaluation")
print("=" * 55)

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
    f"{auto_routed_accuracy:.2%}"
)

print(
    f"\nTotal tickets:            "
    f"{total_tickets}"
)

print(
    f"Auto-routed tickets:      "
    f"{len(auto_routed)}"
)

print(
    f"Manual review tickets:    "
    f"{len(manual_review)}"
)


# ============================================================
# Save
# ============================================================

metrics = {
    "threshold":
        threshold,

    "overall_test_accuracy":
        float(overall_accuracy),

    "coverage":
        float(coverage),

    "manual_review_rate":
        float(manual_review_rate),

    "auto_routed_accuracy":
        float(auto_routed_accuracy),

    "total_tickets":
        int(total_tickets),

    "auto_routed_tickets":
        int(len(auto_routed)),

    "manual_review_tickets":
        int(len(manual_review)),
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


print(
    "\nSaved to:",
    OUTPUT_PATH,
)