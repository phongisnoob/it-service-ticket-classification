import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cnn_data import TicketDataset
from src.data import load_data, split_data
from src.evaluate import calculate_metrics, calculate_calibration_metrics
from src.textcnn import TextCNN

# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CNN_DIR = ROOT_DIR / "artifacts" / "cnn"

METRICS_DIR = ROOT_DIR / "reports" / "metrics"

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load vocabulary
# ============================================================

def main():
    with open(CNN_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(CNN_DIR / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(CNN_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    df = load_data()
    _, val_df, _ = split_data(df, random_state=42)

    label_to_id = {label: index for index, label in enumerate(labels)}
    y_val = [label_to_id[label] for label in val_df["Topic_group"]]

    val_dataset = TicketDataset(
        texts=val_df["Document"],
        labels=y_val,
        vocab=vocab,
        max_length=config["max_length"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

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
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidence.cpu().tolist())

    true_labels = [labels[index] for index in y_val]
    predicted_labels = [labels[index] for index in all_predictions]

    val_metrics = calculate_metrics(true_labels, predicted_labels)
    with open(METRICS_DIR / "cnn_val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=4)

    calib_metrics = calculate_calibration_metrics(
        y_true_labels=true_labels,
        y_pred_labels=predicted_labels,
        y_confidence=all_confidences,
    )
    with open(METRICS_DIR / "cnn_val_calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(calib_metrics, f, indent=4)

    print("\nCNN Validation Metrics")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")
        
    print("\nCalibration Quality:")
    print(f"Brier score: {calib_metrics['brier_score']:.6f}")
    print(f"Expected Calibration Error (ECE): {calib_metrics['expected_calibration_error']:.6f}")

    results = pd.DataFrame({
        "ticket_id": val_df.index.to_numpy(),
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "confidence": all_confidences,
    })
    results["correct"] = results["true_label"] == results["predicted_label"]

    output_path = METRICS_DIR / "cnn_val_predictions.csv"
    results.to_csv(output_path, index=False)

    print(f"Saved validation predictions to: {output_path}")
    print("Validation accuracy:", results["correct"].mean())


if __name__ == "__main__":
    main()