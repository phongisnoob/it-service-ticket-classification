import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_data
from src.paths import FIGURE_DIR, METRICS_DIR

# ============================================================
# Model comparison
# ============================================================


def main() -> None:
    comparison = pd.read_csv(METRICS_DIR / "model_comparison.csv")

    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    plot_data = comparison[["model"] + metrics].set_index("model")

    ax = plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Model Performance Comparison")

    ax.set_ylabel("Score")

    ax.set_ylim(
        0,
        1,
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=200,
    )

    plt.close()

    # ============================================================
    # Threshold trade-off
    # ============================================================

    threshold = pd.read_csv(METRICS_DIR / "baseline_threshold_analysis.csv")

    plt.figure(figsize=(9, 6))

    plt.plot(
        threshold["threshold"],
        threshold["coverage"],
        marker="o",
        label="Coverage",
    )

    plt.plot(
        threshold["threshold"],
        threshold["auto_routed_accuracy"],
        marker="o",
        label="Auto-routed accuracy",
    )

    plt.xlabel("Confidence Threshold")

    plt.ylabel("Score")

    plt.title("Routing Accuracy vs Coverage")

    plt.ylim(
        0,
        1,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "baseline_threshold_tradeoff.png",
        dpi=200,
    )

    plt.close()

    df = load_data()

    class_counts = df["Topic_group"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    class_counts.plot(kind="bar")

    plt.title("Distribution of IT Service Ticket Categories")

    plt.xlabel("Ticket Category")

    plt.ylabel("Number of Tickets")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=200,
    )

    plt.close()

    print(
        "Saved figures to:",
        FIGURE_DIR,
    )


if __name__ == "__main__":
    main()
