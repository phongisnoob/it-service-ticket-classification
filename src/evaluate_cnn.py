import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)
from torch.utils.data import DataLoader

from src.cnn_data import TicketDataset
from src.data import load_data, split_data
from src.evaluate import calculate_calibration_metrics, calculate_metrics
from src.paths import FIGURE_DIR, METRICS_DIR, ROOT_DIR
from src.textcnn import TextCNN
from src.tracking import log_metrics, start_run

CNN_DIR = ROOT_DIR / "artifacts" / "cnn"


def main() -> None:
    with open(CNN_DIR / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(CNN_DIR / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    with open(CNN_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    label_to_id = {label: index for index, label in enumerate(labels)}

    df = load_data()
    _, _, test_df = split_data(df, random_state=42)

    y_test = [label_to_id[label] for label in test_df["Topic_group"]]

    test_dataset = TicketDataset(
        texts=test_df["Document"].tolist(),
        labels=y_test,
        vocab=vocab,
        max_length=int(config["max_length"]),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    all_true, all_pred, all_confidence = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            all_true.extend(targets.tolist())
            all_pred.extend(predictions.cpu().tolist())
            all_confidence.extend(confidence.cpu().tolist())

    metrics = calculate_metrics(all_true, np.asarray(all_pred))

    true_names = [labels[index] for index in all_true]
    pred_names = [labels[index] for index in all_pred]

    calib_metrics = calculate_calibration_metrics(
        y_true_labels=true_names,
        y_pred_labels=np.asarray(pred_names),
        y_confidence=np.asarray(all_confidence),
    )
    metrics.update(calib_metrics)

    print("\nTest Metrics")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report")
    print("=" * 40)
    print(
        classification_report(
            all_true,
            all_pred,
            target_names=labels,
            zero_division=0,
        )
    )

    with open(METRICS_DIR / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with start_run(run_name="evaluate_cnn", model_backend="cnn"):
        log_metrics({"test_" + k: v for k, v in metrics.items()})

    results = pd.DataFrame(
        {
            "ticket_id": test_df.index.to_numpy(),
            "true_label": true_names,
            "predicted_label": pred_names,
            "confidence": all_confidence,
        }
    )
    results["correct"] = results["true_label"] == results["predicted_label"]
    results.to_csv(METRICS_DIR / "cnn_test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        true_names,
        pred_names,
        labels=labels,
        normalize="true",
        xticks_rotation=45,
        ax=ax,
    )
    plt.title("Normalized Confusion Matrix - TextCNN")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cnn_confusion_matrix.png", dpi=200)
    plt.close()

    print("\nSaved predictions and confusion matrix.")


if __name__ == "__main__":
    main()
