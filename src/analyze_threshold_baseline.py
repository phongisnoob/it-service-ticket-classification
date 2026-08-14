"""Threshold selection for the baseline model.

Uses simultaneous one-sided Clopper-Pearson bounds with Bonferroni correction.
Parameters are read from params.yaml so the procedure is fully reproducible.
"""

import json

import pandas as pd
import yaml

from src.hashing import calculate_file_sha256
from src.paths import METRICS_DIR, ROOT_DIR
from src.routing_utils import per_class_accepted_stats, select_threshold

INPUT_PATH = METRICS_DIR / "baseline_val_predictions.csv"
OUTPUT_PATH = METRICS_DIR / "baseline_threshold_analysis.csv"
THRESHOLD_PATH = METRICS_DIR / "baseline_selected_threshold.json"
MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input predictions not found: {INPUT_PATH}")

    with open(ROOT_DIR / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f).get("routing", {})

    target_accuracy: float = float(params.get("target_accuracy", 0.90))
    alpha: float = float(params.get("alpha", 0.05))
    min_samples: int = int(params.get("min_accepted_samples", 50))
    t_start: float = float(params.get("threshold_grid_start", 0.10))
    t_stop: float = float(params.get("threshold_grid_stop", 1.00))
    t_step: float = float(params.get("threshold_grid_step", 0.01))

    results = pd.read_csv(INPUT_PATH)

    selected_row, analysis_df = select_threshold(
        results,
        threshold_start=t_start,
        threshold_stop=t_stop,
        threshold_step=t_step,
        target_accuracy=target_accuracy,
        alpha=alpha,
        min_accepted_samples=min_samples,
    )

    analysis_df.to_csv(OUTPUT_PATH, index=False)
    print("Saved threshold analysis to:", OUTPUT_PATH)

    if selected_row is not None:
        threshold = float(selected_row["threshold"])
        class_stats = per_class_accepted_stats(results, threshold)

        print(f"\nSelected Threshold (Baseline): {threshold:.2f}")
        print(f"  Coverage:  {selected_row['coverage']:.2%}")
        print(f"  Accuracy:  {selected_row['auto_routed_accuracy']:.2%}")
        print(f"  CI lower:  {selected_row['accuracy_ci_lower']:.2%} "
              f"(simultaneous CP, alpha={alpha}, n_candidates={len(analysis_df)})")

        config: dict[str, object] = {
            "threshold": threshold,
            "target_accuracy": target_accuracy,
            "selection_rule": (
                "maximize_coverage_subject_to_simultaneous_clopper_pearson_lower_bound_gte_target"
            ),
            "alpha": alpha,
            "n_candidates": len(analysis_df),
            "min_accepted_samples": min_samples,
            "model_sha256": calculate_file_sha256(MODEL_PATH),
            "validation_coverage": float(selected_row["coverage"]),
            "validation_auto_routed_accuracy": float(selected_row["auto_routed_accuracy"]),
            "validation_accuracy_ci_lower": float(selected_row["accuracy_ci_lower"]),
            "validation_manual_review_rate": float(selected_row["manual_review_rate"]),
            "per_class_stats": class_stats,
        }

        with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print("Saved selected threshold to:", THRESHOLD_PATH)

    else:
        best = analysis_df.loc[analysis_df["auto_routed_accuracy"].idxmax()]
        print(
            f"\nNo threshold met the requirement: "
            f"simultaneous CP lower bound >= {target_accuracy:.0%}."
        )
        print(f"Best point accuracy: {best['auto_routed_accuracy']:.2%} "
              f"at threshold {best['threshold']:.2f}")
        THRESHOLD_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"No threshold has a simultaneous CP lower bound >= {target_accuracy:.0%}. "
            "The previous threshold was removed."
        )


if __name__ == "__main__":
    main()
