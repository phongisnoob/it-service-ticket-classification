from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)

INPUT_PATH = (
    METRICS_DIR
    / "baseline_val_predictions.csv"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "baseline_threshold_analysis.csv"
)

THRESHOLD_PATH = (
    METRICS_DIR
    / "baseline_selected_threshold.json"
)


results = pd.read_csv(
    INPUT_PATH
)


thresholds = np.arange(
    0.40,
    0.91,
    0.05,
)


rows = []



for threshold in thresholds:

    auto_routed_mask = (
        results["confidence"] >= threshold
    )

    routed = results[
        auto_routed_mask
    ]


    # Percentage of tickets automatically routed
    coverage = (
        len(routed)
        / len(results)
    )


    # Accuracy only among automatically routed tickets
    if len(routed) > 0:

        routed_accuracy = (
            routed["correct"].mean()
        )

    else:

        routed_accuracy = 0.0


    manual_review_rate = (
        1 - coverage
    )


    rows.append({
        "threshold":
            round(float(threshold), 2),

        "coverage":
            float(coverage),

        "auto_routed_accuracy":
            float(routed_accuracy),

        "manual_review_rate":
            float(manual_review_rate),
    })



threshold_results = pd.DataFrame(
    rows
)



print("\nThreshold Analysis")
print("=" * 75)

print(
    threshold_results
    .round(4)
    .to_string(
        index=False
    )
)

threshold_results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved threshold analysis to:",
    OUTPUT_PATH,
)


TARGET_ACCURACY = 0.90


candidates = threshold_results[
    threshold_results[
        "auto_routed_accuracy"
    ]
    >= TARGET_ACCURACY
]



if len(candidates) > 0:

    selected = (
        candidates
        .sort_values(
            "coverage",
            ascending=False,
        )
        .iloc[0]
    )


    selected_threshold = float(
        selected["threshold"]
    )


    print("\nSelected Threshold")
    print("=" * 75)

    print(
        f"Threshold:            "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Coverage:             "
        f"{selected['coverage']:.2%}"
    )

    print(
        f"Auto-routed accuracy: "
        f"{selected['auto_routed_accuracy']:.2%}"
    )

    print(
        f"Manual review rate:   "
        f"{selected['manual_review_rate']:.2%}"
    )

    threshold_config = {
        "threshold":
            selected_threshold,

        "target_accuracy":
            TARGET_ACCURACY,

        "validation_coverage":
            float(
                selected["coverage"]
            ),

        "validation_auto_routed_accuracy":
            float(
                selected[
                    "auto_routed_accuracy"
                ]
            ),

        "validation_manual_review_rate":
            float(
                selected[
                    "manual_review_rate"
                ]
            ),
    }


    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            threshold_config,
            f,
            indent=4,
        )


    print(
        "\nSaved selected threshold to:",
        THRESHOLD_PATH,
    )



else:

    best = threshold_results.loc[
        threshold_results[
            "auto_routed_accuracy"
        ].idxmax()
    ]


    print(
        f"\nNo threshold achieved "
        f"{TARGET_ACCURACY:.0%} "
        "auto-routed accuracy."
    )


    print(
        "\nBest validation result"
    )

    print("=" * 75)


    print(
        f"Threshold:            "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Coverage:             "
        f"{best['coverage']:.2%}"
    )

    print(
        f"Auto-routed accuracy: "
        f"{best['auto_routed_accuracy']:.2%}"
    )

    print(
        f"Manual review rate:   "
        f"{best['manual_review_rate']:.2%}"
    )


    print(
        "\nNo production threshold "
        "was saved."
    )