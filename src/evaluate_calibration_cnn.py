"""Held-out calibration evaluation for the TextCNN model.

Evaluates ECE and Brier score on the CALIBRATION partition — the independent
10% hold-out that is never used for training or threshold selection.

See ``evaluate_calibration_baseline.py`` for the full methodology note.
"""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cnn_data import TicketDataset
from src.data import load_calibration_split, load_data
from src.evaluate import calculate_calibration_metrics
from src.paths import METRICS_DIR, ROOT_DIR
from src.textcnn import TextCNN

CNN_DIR = ROOT_DIR / "artifacts" / "cnn"
OUTPUT_PATH = METRICS_DIR / "cnn_calibration_holdout_metrics.json"


def main() -> None:
    with open(CNN_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(CNN_DIR / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(CNN_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    df = load_data()
    calib_df = load_calibration_split(df)

    label_to_id = {label: index for index, label in enumerate(labels)}
    y_calib = [label_to_id[label] for label in calib_df["Topic_group"]]

    calib_dataset = TicketDataset(
        texts=calib_df["Document"].tolist(),
        labels=y_calib,
        vocab=vocab,
        max_length=int(config["max_length"]),
    )

    calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=config["kernel_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    state_dict = torch.load(
        CNN_DIR / "textcnn.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    all_predictions, all_confidences = [], []

    with torch.no_grad():
        for inputs, targets in calib_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidence.cpu().tolist())

    true_labels = [labels[index] for index in y_calib]
    predicted_labels = [labels[index] for index in all_predictions]

    results = pd.DataFrame(
        {
            "ticket_id": calib_df.index.to_numpy(),
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "confidence": all_confidences,
        }
    )
    results["correct"] = results["true_label"] == results["predicted_label"]

    calibration = calculate_calibration_metrics(
        y_true_labels=true_labels,
        y_pred_labels=np.asarray(predicted_labels),
        y_confidence=np.asarray(all_confidences),
    )

    output: dict[str, object] = {
        "partition": "calibration_holdout",
        "n_samples": int(len(results)),
        "accuracy": float(results["correct"].mean()),
        **calibration,
        "note": (
            "Calibration metrics computed on the held-out CALIBRATION partition "
            "(10% of data). This partition was never used for training or "
            "threshold selection."
        ),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Calibration holdout accuracy: {results['correct'].mean():.4f}")
    print(f"Top label Brier score: {calibration['top_label_brier_score']:.6f}")
    print(f"ECE: {calibration['expected_calibration_error']:.6f}")
    print(f"n_samples: {len(results)}")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
