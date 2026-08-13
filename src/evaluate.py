from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true: Any, y_pred: Any) -> Any:

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def print_report(y_true: Any, y_pred: Any) -> Any:

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def calculate_calibration_metrics(
    y_true_labels: Any, y_pred_labels: Any, y_confidence: Any, n_bins: Any = 10
) -> Any:
    """Calculate calibration quality metrics for confidence-based routing.

    Parameters
    ----------
    y_true_labels : array-like
        True class labels.
    y_pred_labels : array-like
        Predicted class labels.
    y_confidence : array-like
        Maximum predicted probability (confidence) for each sample.
    n_bins : int
        Number of bins for Expected Calibration Error.

    Returns
    -------
    dict
        Brier score (lower is better) and Expected Calibration Error.
    """
    y_true_arr = np.asarray(y_true_labels)
    y_pred_arr = np.asarray(y_pred_labels)
    confidence = np.asarray(y_confidence, dtype=float)
    correct = (y_true_arr == y_pred_arr).astype(float)

    # Brier score: mean squared error between confidence and correctness
    brier = float(np.mean((confidence - correct) ** 2))

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidence)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

    return {
        "top_label_brier_score": round(brier, 6),
        "expected_calibration_error": round(float(ece), 6),
    }
