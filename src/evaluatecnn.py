import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader

from src.data import load_data, split_data
from src.evaluate import calculate_metrics
from src.cnn_data import TicketDataset
from src.textcnn import TextCNN


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    ROOT_DIR
    / "artifacts"
    / "cnn"
)

METRICS_DIR = (
    ROOT_DIR
    / "reports"
    / "metrics"
)

FIGURE_DIR = (
    ROOT_DIR
    / "reports"
    / "figures"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load metadata
# ============================================================

with open(
    ARTIFACT_DIR / "vocab.json",
    encoding="utf-8",
) as f:

    vocab = json.load(f)


with open(
    ARTIFACT_DIR / "labels.json",
    encoding="utf-8",
) as f:

    labels = json.load(f)


with open(
    ARTIFACT_DIR / "config.json",
    encoding="utf-8",
) as f:

    config = json.load(f)


label_to_id = {
    label: index
    for index, label in enumerate(labels)
}


# ============================================================
# Recreate test split
# ============================================================

df = load_data()

_, _, test_df = split_data(
    df,
    random_state=42,
)

y_test = [
    label_to_id[label]
    for label
    in test_df["Topic_group"]
]


test_dataset = TicketDataset(
    texts=test_df["Document"],
    labels=y_test,
    vocab=vocab,
    max_length=config["max_length"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Recreate architecture
# ============================================================

model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=config["embedding_dim"],
    num_filters=config["num_filters"],
    kernel_sizes=config["kernel_sizes"],
    num_classes=config["num_classes"],
    dropout=config["dropout"],
)

model = model.to(device)


# ============================================================
# Load weights
# ============================================================

state_dict = torch.load(
    ARTIFACT_DIR / "textcnn.pt",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# Predictions
# ============================================================

all_true = []
all_pred = []
all_confidence = []


with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(device)

        logits = model(inputs)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        all_true.extend(
            targets.tolist()
        )

        all_pred.extend(
            predictions.cpu().tolist()
        )

        all_confidence.extend(
            confidence.cpu().tolist()
        )


# ============================================================
# Metrics
# ============================================================

metrics = calculate_metrics(
    all_true,
    all_pred,
)

print("\nTest Metrics")
print("=" * 40)

for key, value in metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# Classification report
# ============================================================

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


# ============================================================
# Save metrics
# ============================================================

with open(
    METRICS_DIR / "cnn_metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# Convert integer predictions back to names
# ============================================================

true_names = [
    labels[index]
    for index in all_true
]

pred_names = [
    labels[index]
    for index in all_pred
]


# ============================================================
# Save predictions for error analysis
# ============================================================

results = pd.DataFrame({
    "Document":
        test_df["Document"].values,

    "true_label":
        true_names,

    "predicted_label":
        pred_names,

    "confidence":
        all_confidence,
})


results["correct"] = (
    results["true_label"]
    ==
    results["predicted_label"]
)


results.to_csv(
    METRICS_DIR
    / "cnn_test_predictions.csv",
    index=False,
)


# ============================================================
# Confusion matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ConfusionMatrixDisplay.from_predictions(
    true_names,
    pred_names,
    labels=labels,
    normalize="true",
    xticks_rotation=45,
    ax=ax,
)

plt.title(
    "Normalized Confusion Matrix - TextCNN"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "cnn_confusion_matrix.png",
    dpi=200,
)

plt.close()


print(
    "\nSaved predictions and confusion matrix."
)