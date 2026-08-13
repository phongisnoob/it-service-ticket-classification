import pandas as pd

from src.paths import METRICS_DIR


def main():
    results = pd.read_csv(METRICS_DIR / "cnn_test_predictions.csv")

    errors = results[not results["correct"]].copy()

    # ============================================================
    # Most common confusion pairs
    # ============================================================

    confusions = (
        errors.groupby(
            [
                "true_label",
                "predicted_label",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    confusions.to_csv(
        METRICS_DIR / "error_pairs.csv",
        index=False,
    )

    print("\nMost Common Errors")
    print("=" * 70)

    print(confusions.head(15).to_string(index=False))

    # ============================================================
    # High-confidence errors
    # ============================================================

    high_confidence_errors = errors[errors["confidence"] >= 0.80].sort_values(
        "confidence",
        ascending=False,
    )

    high_confidence_errors.to_csv(
        METRICS_DIR / "high_confidence_errors.csv",
        index=False,
    )

    print(
        "\nHigh-confidence errors:",
        len(high_confidence_errors),
    )


if __name__ == "__main__":
    main()
