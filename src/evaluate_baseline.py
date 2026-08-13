import json

import joblib

from src.data import (
    load_data,
    split_data,
)
from src.evaluate import (
    calculate_metrics,
    print_report,
)
from src.paths import ARTIFACT_DIR, METRICS_DIR

MODEL_PATH = ARTIFACT_DIR / "baseline.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    _, _, test_df = split_data(
        load_data(),
        random_state=42,
    )

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

    print(
        "Saved test metrics to:",
        METRICS_DIR / "baseline_metrics.json",
    )


if __name__ == "__main__":
    main()
