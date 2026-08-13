import hashlib
import json
import typing
from typing import Any

import joblib
import numpy as np
import torch

from src.cnn_data import encode_text
from src.paths import (
    ARTIFACT_DIR,
    BASELINE_THRESHOLD_PATH,
    CNN_THRESHOLD_PATH,
    MODEL_SELECTION_PATH,
)
from src.textcnn import TextCNN


def get_selected_backend() -> str:
    if not MODEL_SELECTION_PATH.exists():
        raise FileNotFoundError(
            "model_selection.json not found. Run python -m src.select_model first."
        )

    with open(
        MODEL_SELECTION_PATH,
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    backend = selection.get("selected_backend")

    if backend not in {
        "baseline",
        "cnn",
    }:
        raise RuntimeError("Invalid selected_backend in model_selection.json.")

    return backend  # type: ignore


def calculate_sha256(path: typing.Any) -> typing.Any:
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_threshold(
    threshold_path: Any,
    model_path: Any,
) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with open(
        threshold_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    expected_hash = config.get("model_sha256")

    if not expected_hash:
        raise RuntimeError(
            "Threshold config does not contain a model SHA-256 hash. Run threshold selection again."
        )

    actual_hash = calculate_sha256(model_path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "The model and threshold were generated "
            "from different model versions. "
            "Run threshold selection again."
        )

    threshold = float(config["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold


def format_prediction(
    labels: Any,
    probabilities: Any,
    threshold: Any,
) -> Any:

    order = np.argsort(probabilities)[::-1]

    best_index = int(order[0])

    confidence = float(probabilities[best_index])

    top_3 = [
        {
            "category": str(labels[index]),
            "probability": float(probabilities[index]),
        }
        for index in order[:3]
    ]

    return {
        "category": str(labels[best_index]),
        "confidence": confidence,
        "threshold": float(threshold),
        "needs_manual_review": confidence < threshold,
        "top_3": top_3,
    }


# Logistic Regression


class BaselinePredictor:
    def __init__(self) -> None:

        model_path = ARTIFACT_DIR / "baseline.joblib"

        self.backend = "baseline"

        self.threshold = load_threshold(
            BASELINE_THRESHOLD_PATH,
            model_path,
        )

        self.model_sha256 = calculate_sha256(model_path)

        self.model = joblib.load(model_path)

        if hasattr(self.model, "classes_"):
            self.labels = self.model.classes_
        elif hasattr(self.model, "named_steps"):
            self.labels = self.model.named_steps["classifier"].classes_
        else:
            raise AttributeError("Loaded model has no classes_ attribute")

    def predict(
        self,
        text: Any,
    ) -> Any:

        probabilities = self.model.predict_proba([text])[0]

        return format_prediction(
            self.labels,
            probabilities,
            self.threshold,
        )


# CNN


class CNNPredictor:
    def __init__(self) -> None:

        cnn_dir = ARTIFACT_DIR / "cnn"

        self.backend = "cnn"

        weights_path = cnn_dir / "textcnn.pt"
        vocab_path = cnn_dir / "vocab.json"
        manifest_path = cnn_dir / "artifact_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("CNN artifact_manifest.json is missing.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for filename, expected_hash in manifest.items():
            filepath = cnn_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing CNN artifact: {filename}")
            actual_hash = calculate_sha256(filepath)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Hash mismatch for CNN artifact {filename}. Expected {expected_hash}, got {actual_hash}."
                )

        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(
            cnn_dir / "labels.json",
            encoding="utf-8",
        ) as f:
            self.labels = json.load(f)

        with open(
            cnn_dir / "config.json",
            encoding="utf-8",
        ) as f:
            self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=self.config["embedding_dim"],
            num_filters=self.config["num_filters"],
            kernel_sizes=self.config["kernel_sizes"],
            num_classes=self.config["num_classes"],
            dropout=self.config["dropout"],
        )

        state_dict = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        self.model.eval()

        self.threshold = load_threshold(
            CNN_THRESHOLD_PATH,
            weights_path,
        )

        self.model_sha256 = calculate_sha256(weights_path)

    def predict(
        self,
        text: Any,
    ) -> Any:

        token_ids = encode_text(
            text,
            self.vocab,
            max_length=self.config["max_length"],
        )

        x = torch.tensor(
            [token_ids],
            dtype=torch.long,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )[0]
                .cpu()
                .numpy()
            )

        return format_prediction(
            self.labels,
            probabilities,
            self.threshold,
        )


# Factory


def get_predictor(
    backend: typing.Any = "auto",
) -> typing.Any:
    if backend == "auto":
        backend = get_selected_backend()

    if backend == "baseline":
        return BaselinePredictor()

    if backend == "cnn":
        return CNNPredictor()

    raise ValueError(f"Unknown backend: {backend}")
