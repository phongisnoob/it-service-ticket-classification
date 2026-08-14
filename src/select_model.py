import json
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.paths import METRICS_DIR

TARGET_ACCURACY = 0.90


class CandidateResult(TypedDict):
    backend: str
    validation_accuracy: float
    validation_macro_f1: float
    threshold: float
    validation_coverage: float
    validation_auto_routed_accuracy: float
    validation_accuracy_ci_lower: float
    model_sha256: str


class ModelPaths(TypedDict):
    predictions: Path
    threshold: Path


MODEL_CONFIGS: dict[str, ModelPaths] = {
    "baseline": {
        "predictions": METRICS_DIR / "baseline_val_predictions.csv",
        "threshold": METRICS_DIR / "baseline_selected_threshold.json",
    },
    "cnn": {
        "predictions": METRICS_DIR / "cnn_val_predictions.csv",
        "threshold": METRICS_DIR / "selected_threshold.json",
    },
}


def load_candidate(backend: str, paths: ModelPaths) -> CandidateResult:
    predictions_path = paths["predictions"]
    threshold_path = paths["threshold"]

    if not predictions_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {predictions_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold config not found: {threshold_path}")

    predictions = pd.read_csv(predictions_path)
    with open(threshold_path, encoding="utf-8") as f:
        threshold = json.load(f)

    ci_lower: float | None = threshold.get("validation_accuracy_ci_lower")
    if ci_lower is None:
        raise RuntimeError(
            f"{backend} threshold artifact does not contain validation_accuracy_ci_lower. "
            "Run threshold analysis again."
        )

    return {
        "backend": backend,
        "validation_accuracy": float(
            accuracy_score(predictions["true_label"], predictions["predicted_label"])
        ),
        "validation_macro_f1": float(
            f1_score(
                predictions["true_label"],
                predictions["predicted_label"],
                average="macro",
                zero_division=0,
            )
        ),
        "threshold": float(threshold["threshold"]),
        "validation_coverage": float(threshold["validation_coverage"]),
        "validation_auto_routed_accuracy": float(threshold["validation_auto_routed_accuracy"]),
        "validation_accuracy_ci_lower": float(ci_lower),
        "model_sha256": threshold["model_sha256"],
    }


def main() -> None:
    candidates = [load_candidate(backend, paths) for backend, paths in MODEL_CONFIGS.items()]

    eligible = [c for c in candidates if c["validation_accuracy_ci_lower"] >= TARGET_ACCURACY]

    if not eligible:
        raise RuntimeError(
            "No model satisfies the routing requirement: "
            f"lower 95% accuracy CI >= {TARGET_ACCURACY:.0%}."
        )

    winner = max(
        eligible,
        key=lambda c: (c["validation_coverage"], c["validation_macro_f1"]),
    )

    output: dict[str, object] = {
        "selection_dataset": "validation",
        "target_accuracy": TARGET_ACCURACY,
        "selection_rule": (
            "Highest validation coverage subject to "
            "lower 95% auto-routing accuracy CI meeting the target; Macro F1 as tie-breaker"
        ),
        "selected_backend": winner["backend"],
        "selected_threshold": winner["threshold"],
        "selected_model_sha256": winner["model_sha256"],
        "candidates": candidates,
    }

    output_path = METRICS_DIR / "model_selection.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Selected backend: {winner['backend']}")
    print(f"Selected threshold: {winner['threshold']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
