import json

import joblib

from src.data import (
    load_data,
    load_splits,
)
from src.evaluate import (
    calculate_metrics,
    print_report,
)
from src.paths import ARTIFACT_DIR, METRICS_DIR
from src.tracking import log_metrics, start_run

MODEL_PATH = ARTIFACT_DIR / "baseline.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    test_df = load_splits(load_data()).test

    predictions = model.predict(test_df["Document"])

    metrics = calculate_metrics(
        test_df["Topic_group"],
        predictions,
    )

    print_report(
        test_df["Topic_group"],
        predictions,
    )

    with open(
        METRICS_DIR / "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    with start_run(run_name="evaluate_baseline", model_backend="baseline"):
        log_metrics({"test_" + k: v for k, v in metrics.items()})

    print("Saved test metrics to:", METRICS_DIR / "baseline_metrics.json")


if __name__ == "__main__":
    main()
