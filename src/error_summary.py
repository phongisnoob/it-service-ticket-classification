"""Error analysis for model predictions.

Produces:
- reports/metrics/error_pairs.csv   — most common (true, predicted) confusion pairs
- reports/metrics/high_confidence_errors.csv — errors where confidence >= 0.80

High-confidence errors are a critical failure category: the model was certain
but wrong.  These deserve manual inspection to identify systematic failures,
label noise, or semantic ambiguity in the taxonomy.

Pass ``--model baseline`` or ``--model cnn`` to select which predictions file
to analyse.  Defaults to CNN test predictions.
"""

import argparse

import pandas as pd

from src.paths import METRICS_DIR

PREDICTIONS_FILES = {
    "cnn": METRICS_DIR / "cnn_test_predictions.csv",
    "baseline": METRICS_DIR / "baseline_tune_predictions.csv",
}

HIGH_CONFIDENCE_THRESHOLD = 0.80
TOP_N_ERRORS = 15


def run_error_analysis(predictions_path: object) -> None:
    import pathlib
    path = pathlib.Path(str(predictions_path))
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    results = pd.read_csv(path)

    required_cols = {"true_label", "predicted_label", "correct", "confidence"}
    missing_cols = required_cols - set(results.columns)
    if missing_cols:
        raise ValueError(f"Predictions file missing columns: {missing_cols}")

    # BUG-FIX: was `results[not results["correct"]]` which raises ValueError
    # on a pandas Series. Correct form is `~results["correct"]`.
    errors = results[~results["correct"]].copy()

    # ================================================================
    # Most common confusion pairs
    # ================================================================

    confusions = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    confusions.to_csv(METRICS_DIR / "error_pairs.csv", index=False)

    print("\nMost Common Errors")
    print("=" * 70)
    print(confusions.head(TOP_N_ERRORS).to_string(index=False))

    # ================================================================
    # High-confidence errors (critical failure category)
    # ================================================================

    high_confidence_errors = (
        errors[errors["confidence"] >= HIGH_CONFIDENCE_THRESHOLD]
        .sort_values("confidence", ascending=False)
    )

    high_confidence_errors.to_csv(METRICS_DIR / "high_confidence_errors.csv", index=False)

    n_errors = len(errors)
    n_total = len(results)
    n_hce = len(high_confidence_errors)

    print(f"\nTotal predictions:        {n_total}")
    print(f"Errors:                   {n_errors} ({n_errors / n_total:.1%})")
    print(f"High-confidence errors:   {n_hce} "
          f"(confidence >= {HIGH_CONFIDENCE_THRESHOLD:.0%})")
    if n_errors > 0:
        print(f"  HCE fraction of errors: {n_hce / n_errors:.1%}")

    print(f"\nSaved confusion pairs to:       {METRICS_DIR / 'error_pairs.csv'}")
    print(f"Saved high-confidence errors to: {METRICS_DIR / 'high_confidence_errors.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis on model predictions.")
    parser.add_argument(
        "--model",
        choices=list(PREDICTIONS_FILES.keys()),
        default="cnn",
        help="Which model's predictions to analyse (default: cnn)",
    )
    args = parser.parse_args()

    predictions_path = PREDICTIONS_FILES[args.model]
    run_error_analysis(predictions_path)


if __name__ == "__main__":
    main()
