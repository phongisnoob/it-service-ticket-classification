import json
from pathlib import Path

import joblib

from src.data import (
    load_data,
    split_data,
)
from src.evaluate import (
    calculate_metrics,
    print_report,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"

OUTPUT_PATH = ROOT_DIR / "reports" / "metrics" / "baseline_metrics.json"


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
        OUTPUT_PATH,
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
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
