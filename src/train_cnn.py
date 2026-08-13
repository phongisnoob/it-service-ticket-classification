import datetime
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.cnn_data import TicketDataset, build_vocab
from src.data import DATA_PATH, calculate_file_sha256, load_data, split_data
from src.textcnn import TextCNN

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "cnn"

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
MAX_LENGTH = 100
MAX_VOCAB_SIZE = 30000
MIN_FREQ = 2

EMBEDDING_DIM = 128
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]
DROPOUT = 0.3

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 15
PATIENCE = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cnn():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = load_data()
    train_df, val_df, _ = split_data(df, random_state=SEED)

    print("Train:", len(train_df))
    print("Validation:", len(val_df))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Topic_group"])
    y_val = label_encoder.transform(val_df["Topic_group"])
    num_classes = len(label_encoder.classes_)

    print("Classes:", label_encoder.classes_)

    vocab = build_vocab(
        train_df["Document"],
        min_freq=MIN_FREQ,
        max_vocab_size=MAX_VOCAB_SIZE,
    )
    print("Vocabulary size:", len(vocab))

    train_dataset = TicketDataset(
        texts=train_df["Document"],
        labels=y_train,
        vocab=vocab,
        max_length=MAX_LENGTH,
    )
    val_dataset = TicketDataset(
        texts=val_df["Document"],
        labels=y_val,
        vocab=vocab,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = len(y_train) / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train_one_epoch():
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(train_dataset)

    def validate():
        model.eval()
        total_loss = 0
        all_targets, all_predictions = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(logits, dim=1)
                all_targets.extend(targets.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())
        average_loss = total_loss / len(val_dataset)
        macro_f1 = f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        )
        return average_loss, macro_f1

    with open(ARTIFACT_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with open(ARTIFACT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

    config = {
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "dropout": DROPOUT,
        "num_classes": num_classes,
    }

    with open(ARTIFACT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val_f1 = -1
    epochs_without_improvement = 0
    model_path = ARTIFACT_DIR / "textcnn.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch()
        val_loss, val_f1 = validate()
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print("  -> Saved new best model")
        else:
            epochs_without_improvement += 1
            print("  -> No improvement:", epochs_without_improvement)

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping.")
            break

    model_sha256 = calculate_file_sha256(model_path)
    dataset_sha256 = calculate_file_sha256(DATA_PATH)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None

    metadata = {
        "model_backend": "cnn",
        "pytorch_version": torch.__version__,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "random_seed": SEED,
        "hyperparameters": config,
        "best_val_macro_f1": best_val_f1,
    }

    with open(ARTIFACT_DIR / "cnn_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nBest validation Macro-F1:", round(best_val_f1, 4))
    print(f"Best model saved to {model_path} (SHA-256: {model_sha256[:12]}...)")


def main():
    train_cnn()


if __name__ == "__main__":
    main()