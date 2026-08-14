import datetime
import json
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.cnn_data import TicketDataset, build_vocab
from src.data import load_data, split_data
from src.hashing import calculate_file_sha256
from src.paths import ARTIFACT_DIR, DATA_PATH, ROOT_DIR
from src.textcnn import TextCNN
from src.tracking import log_artifact, log_dict_as_artifact, log_metrics, log_params, start_run

CNN_DIR = ARTIFACT_DIR / "cnn"

with open(ROOT_DIR / "params.yaml", "r") as f:
    cnn_params = yaml.safe_load(f)["cnn"]

SEED = cnn_params["seed"]
MAX_VOCAB_SIZE = cnn_params["max_vocab_size"]
MIN_FREQ = cnn_params["min_freq"]

EMBEDDING_DIM = cnn_params["embedding_dim"]
NUM_FILTERS = cnn_params["num_filters"]
KERNEL_SIZES = cnn_params["kernel_sizes"]
DROPOUT = cnn_params["dropout"]

BATCH_SIZE = cnn_params["batch_size"]
LEARNING_RATE = cnn_params["learning_rate"]
MAX_EPOCHS = cnn_params["max_epochs"]
PATIENCE = cnn_params["patience"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: TextCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    device: torch.device,
    dataset_size: int,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / dataset_size


def evaluate_on_val(
    model: TextCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    dataset_size: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * inputs.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())

    macro_f1: float = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    return total_loss / dataset_size, macro_f1


def train_cnn() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = load_data()
    train_df, val_df, _ = split_data(df, random_state=SEED)
    print(f"Train: {len(train_df)}  Validation: {len(val_df)}")

    label_encoder = LabelEncoder()
    y_train: np.ndarray = label_encoder.fit_transform(train_df["Topic_group"])
    y_val: np.ndarray = label_encoder.transform(val_df["Topic_group"])
    num_classes = len(label_encoder.classes_)
    print("Classes:", label_encoder.classes_)

    # Build the vocabulary from training data only to avoid leaking
    # validation/test token frequencies.
    vocab = build_vocab(train_df["Document"].tolist(), min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
    print("Vocabulary size:", len(vocab))

    token_lengths = train_df["Document"].apply(lambda x: len(str(x).split()))
    p50 = float(np.percentile(token_lengths, 50))
    p90 = float(np.percentile(token_lengths, 90))
    p95 = float(np.percentile(token_lengths, 95))
    p99 = float(np.percentile(token_lengths, 99))
    max_length = int(p95)
    training_truncation_rate = float((token_lengths > max_length).mean())
    print(f"Selected max_length: {max_length} (P95)")

    train_dataset = TicketDataset(
        texts=train_df["Document"].tolist(),
        labels=y_train.tolist(),
        vocab=vocab,
        max_length=max_length,
    )
    val_dataset = TicketDataset(
        texts=val_df["Document"].tolist(),
        labels=y_val.tolist(),
        vocab=vocab,
        max_length=max_length,
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = torch.tensor(
        len(y_train) / (num_classes * class_counts), dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CNN_DIR.mkdir(parents=True, exist_ok=True)

    config: dict[str, object] = {
        "max_length": max_length,
        "embedding_dim": EMBEDDING_DIM,
        "num_filters": NUM_FILTERS,
        "kernel_sizes": KERNEL_SIZES,
        "dropout": DROPOUT,
        "num_classes": num_classes,
        "training_truncation_rate": training_truncation_rate,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }

    with open(CNN_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(CNN_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)
    with open(CNN_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    model_path = CNN_DIR / "textcnn.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, len(train_dataset)
        )
        val_loss, val_f1 = evaluate_on_val(model, val_loader, criterion, device, len(val_dataset))
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
            print(f"  -> No improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping.")
            break

    model_sha256 = calculate_file_sha256(model_path)

    try:
        git_commit: str | None = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None

    metadata: dict[str, object] = {
        "model_backend": "cnn",
        "pytorch_version": torch.__version__,
        "dataset_sha256": calculate_file_sha256(DATA_PATH),
        "model_sha256": model_sha256,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "random_seed": SEED,
        "hyperparameters": config,
        "best_val_macro_f1": best_val_f1,
    }

    with open(CNN_DIR / "cnn_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    manifest: dict[str, str] = {
        "textcnn.pt": calculate_file_sha256(CNN_DIR / "textcnn.pt"),
        "vocab.json": calculate_file_sha256(CNN_DIR / "vocab.json"),
        "labels.json": calculate_file_sha256(CNN_DIR / "labels.json"),
        "config.json": calculate_file_sha256(CNN_DIR / "config.json"),
    }
    with open(CNN_DIR / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    with start_run(run_name="train_cnn", model_backend="cnn"):
        log_params(cnn_params)
        log_metrics({"best_val_macro_f1": best_val_f1})
        log_artifact(str(model_path), "artifacts/cnn")
        log_dict_as_artifact(metadata, "cnn_metadata.json")

    print(f"\nBest validation Macro-F1: {best_val_f1:.4f}")
    print(f"Best model saved to {model_path} (SHA-256: {model_sha256[:12]}...)")


def main() -> None:
    train_cnn()


if __name__ == "__main__":
    main()
