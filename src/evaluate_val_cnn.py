import json
from pathlib import Path

import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data import load_data, split_data
from src.cnn_data import TicketDataset
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

with open(
    CNN_DIR / "vocab.json",
    "r",
    encoding="utf-8",
) as f:
    vocab = json.load(f)


# ============================================================
# Load labels
# ============================================================

with open(
    CNN_DIR / "labels.json",
    "r",
    encoding="utf-8",
) as f:
    labels = json.load(f)


# ============================================================
# Load config
# ============================================================

with open(
    CNN_DIR / "config.json",
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)


# ============================================================
# Recreate validation split
# ============================================================

df = load_data()

train_df, val_df, test_df = split_data(
    df,
    random_state=42,
)


# ============================================================
# Convert labels to integer IDs
# ============================================================

label_to_id = {
    label: index
    for index, label in enumerate(labels)
}

y_val = [
    label_to_id[label]
    for label in val_df["Topic_group"]
]


# ============================================================
# Validation Dataset
# ============================================================

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


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# Recreate CNN
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
# Load best checkpoint
# ============================================================

state_dict = torch.load(
    CNN_DIR / "textcnn.pt",
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)

model.eval()


# ============================================================
# Validation inference
# ============================================================

all_predictions = []
all_confidences = []


with torch.no_grad():

    for inputs, targets in val_loader:

        inputs = inputs.to(device)

        logits = model(inputs)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predictions = probabilities.max(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_confidences.extend(
            confidence.cpu().tolist()
        )


# ============================================================
# Convert IDs back to category names
# ============================================================

true_labels = [
    labels[index]
    for index in y_val
]

predicted_labels = [
    labels[index]
    for index in all_predictions
]


# ============================================================
# Save results
# ============================================================

results = pd.DataFrame({
    "true_label":
        true_labels,

    "predicted_label":
        predicted_labels,

    "confidence":
        all_confidences,
})


results["correct"] = (
    results["true_label"]
    ==
    results["predicted_label"]
)


output_path = (
    METRICS_DIR
    / "cnn_val_predictions.csv"
)

results.to_csv(
    output_path,
    index=False,
)


print(
    f"Saved validation predictions to: {output_path}"
)

print(
    "Validation accuracy:",
    results["correct"].mean()
)