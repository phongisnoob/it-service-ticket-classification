import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true: "pd.Series", y_pred: "np.ndarray") -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_report(y_true: "pd.Series", y_pred: "np.ndarray") -> None:
    print(classification_report(y_true, y_pred, zero_division=0))


def calculate_calibration_metrics(
    y_true_labels: "pd.Series",
    y_pred_labels: "np.ndarray",
    y_confidence: "np.ndarray",
    n_bins: int = 10,
) -> dict[str, float]:
    """Brier score and Expected Calibration Error for the confidence routing layer.

    These measure how well the model's confidence scores correlate with actual correctness,
    which matters when we use confidence to decide whether to auto-route or escalate.
    """
    y_true_arr = np.asarray(y_true_labels)
    y_pred_arr = np.asarray(y_pred_labels)
    confidence = np.asarray(y_confidence, dtype=float)
    correct = (y_true_arr == y_pred_arr).astype(float)

    brier = float(np.mean((confidence - correct) ** 2))

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidence)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == n_bins - 1 else confidence < hi)
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        ece += (bin_size / total) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )

    return {
        "top_label_brier_score": round(brier, 6),
        "expected_calibration_error": round(ece, 6),
    }


try:
    import pandas as pd  # noqa: F401
except ImportError:
    pass
