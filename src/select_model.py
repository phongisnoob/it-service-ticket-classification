import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)


MODEL_CONFIGS = {
    "baseline": {
        "predictions":
            METRICS_DIR
            / "baseline_val_predictions.csv",

        "threshold":
            METRICS_DIR
            / "baseline_selected_threshold.json",
    },

    "cnn": {
        "predictions":
            METRICS_DIR
            / "cnn_val_predictions.csv",

        "threshold":
            METRICS_DIR
            / "selected_threshold.json",
    },
}


def load_candidate(
    backend,
    paths,
):
    predictions = pd.read_csv(
        paths["predictions"]
    )

    with open(
        paths["threshold"],
        "r",
        encoding="utf-8",
    ) as file:
        threshold = json.load(file)

    return {
        "backend":
            backend,

        "validation_accuracy":
            float(
                accuracy_score(
                    predictions["true_label"],
                    predictions["predicted_label"],
                )
            ),

        "validation_macro_f1":
            float(
                f1_score(
                    predictions["true_label"],
                    predictions["predicted_label"],
                    average="macro",
                    zero_division=0,
                )
            ),

        "validation_coverage":
            float(
                threshold[
                    "validation_coverage"
                ]
            ),

        "validation_auto_routed_accuracy":
            float(
                threshold[
                    "validation_auto_routed_accuracy"
                ]
            ),
    }


def main():
    candidates = [
        load_candidate(
            backend,
            paths,
        )
        for backend, paths
        in MODEL_CONFIGS.items()
    ]

    eligible = [
        candidate
        for candidate in candidates
        if candidate[
            "validation_auto_routed_accuracy"
        ] >= 0.90
    ]

    if not eligible:
        raise RuntimeError(
            "No model meets the routing target."
        )

    winner = max(
        eligible,
        key=lambda candidate: (
            candidate[
                "validation_coverage"
            ],
            candidate[
                "validation_macro_f1"
            ],
        ),
    )

    output = {
        "selection_dataset":
            "validation",

        "selection_rule":
            (
                "Highest coverage with at least "
                "90% auto-routed accuracy; "
                "Macro F1 as tie-breaker"
            ),

        "selected_backend":
            winner["backend"],

        "candidates":
            candidates,
    }

    output_path = (
        METRICS_DIR
        / "model_selection.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        "Selected backend:",
        winner["backend"],
    )

    print(
        "Saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
