import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT_DIR / "reports" / "metrics"
CNN_MODEL_PATH = ROOT_DIR / "artifacts" / "cnn" / "textcnn.pt"


BASELINE_OUTPUTS = [
    "baseline_val_predictions.csv",
    "baseline_threshold_analysis.csv",
    "baseline_selected_threshold.json",
    "baseline_metrics.json",
    "baseline_routing_metrics.json",
]

CNN_OUTPUTS = [
    "cnn_val_predictions.csv",
    "threshold_analysis.csv",
    "selected_threshold.json",
    "cnn_metrics.json",
    "cnn_test_predictions.csv",
    "routing_metrics.json",
]

COMMON_OUTPUTS = [
    "model_comparison.csv",
    "model_selection.json",
]


def remove_stale_outputs(names: list[str]) -> None:
    """Remove generated metrics before rebuilding them."""
    for name in names:
        path = METRICS_DIR / name

        if path.exists():
            path.unlink()
            print(f"Removed stale artifact: {path}")


def run_module(module: str) -> None:
    """Run a Python module and fail immediately on error."""
    print("\n" + "=" * 72)
    print(f"Running: python -m {module}")
    print("=" * 72)

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        cwd=ROOT_DIR,
        check=True,
    )


def main() -> None:
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nRegenerating baseline evaluation artifacts...")

    remove_stale_outputs(
        BASELINE_OUTPUTS
        + COMMON_OUTPUTS
    )

    run_module("src.evaluate_val_baseline")
    run_module("src.analyze_threshold_baseline")
    run_module("src.evaluate_baseline")
    run_module("src.evaluate_routing_baseline")

    if CNN_MODEL_PATH.exists():
        print("\nCNN artifact detected.")
        print("Regenerating CNN evaluation artifacts...")

        remove_stale_outputs(
            CNN_OUTPUTS
        )

        run_module("src.evaluate_val_cnn")
        run_module("src.analyze_threshold_cnn")
        run_module("src.evaluate_cnn")
        run_module("src.evaluate_routing")

        run_module("src.compare_models")
        run_module("src.select_model")
    else:
        print(
            "\nCNN model not found. "
            "Skipping CNN comparison/model selection."
        )

    print("\nEvaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()
