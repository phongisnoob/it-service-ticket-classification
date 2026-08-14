import json
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import torch

from src.cnn_data import encode_text
from src.hashing import calculate_file_sha256
from src.paths import (
    ARTIFACT_DIR,
    BASELINE_THRESHOLD_PATH,
    CNN_THRESHOLD_PATH,
    MODEL_SELECTION_PATH,
)
from src.textcnn import TextCNN


class TopPrediction(TypedDict):
    category: str
    probability: float


class PredictionResult(TypedDict):
    category: str
    confidence: float
    threshold: float
    needs_manual_review: bool
    top_3: list[TopPrediction]


def get_selected_backend() -> str:
    if not MODEL_SELECTION_PATH.exists():
        raise FileNotFoundError(
            "model_selection.json not found. Run python -m src.select_model first."
        )
    with open(MODEL_SELECTION_PATH, encoding="utf-8") as f:
        selection = json.load(f)

    backend: str = selection.get("selected_backend", "")
    if backend not in {"baseline", "cnn"}:
        raise RuntimeError("Invalid selected_backend in model_selection.json.")
    return backend


def load_threshold(threshold_path: Path, model_path: Path) -> float:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with open(threshold_path, encoding="utf-8") as f:
        config = json.load(f)

    expected_hash: str = config.get("model_sha256", "")
    if not expected_hash:
        raise RuntimeError(
            "Threshold config does not contain a model SHA-256 hash. Run threshold selection again."
        )

    actual_hash = calculate_file_sha256(model_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "The model and threshold were generated from different model versions. "
            "Run threshold selection again."
        )

    threshold = float(config["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return threshold


# Expose calculate_sha256 for tests that import it directly from this module.
calculate_sha256 = calculate_file_sha256


def format_prediction(
    labels: "np.ndarray | list[str]",
    probabilities: "np.ndarray",
    threshold: float,
) -> PredictionResult:
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    confidence = float(probabilities[best_index])

    top_3: list[TopPrediction] = [
        {"category": str(labels[i]), "probability": float(probabilities[i])}
        for i in order[:3]
    ]

    return {
        "category": str(labels[best_index]),
        "confidence": confidence,
        "threshold": threshold,
        "needs_manual_review": confidence < threshold,
        "top_3": top_3,
    }


class BaselinePredictor:
    backend = "baseline"

    def __init__(self) -> None:
        model_path = ARTIFACT_DIR / "baseline.joblib"
        self.threshold = load_threshold(BASELINE_THRESHOLD_PATH, model_path)
        self.model_sha256 = calculate_file_sha256(model_path)
        self.model = joblib.load(model_path)

        if hasattr(self.model, "classes_"):
            self.labels: "np.ndarray" = self.model.classes_
        elif hasattr(self.model, "named_steps"):
            self.labels = self.model.named_steps["classifier"].classes_
        else:
            raise AttributeError("Loaded model has no classes_ attribute")

    def predict(self, text: str) -> PredictionResult:
        probabilities: "np.ndarray" = self.model.predict_proba([text])[0]
        return format_prediction(self.labels, probabilities, self.threshold)


class CNNPredictor:
    backend = "cnn"

    def __init__(self) -> None:
        cnn_dir = ARTIFACT_DIR / "cnn"
        weights_path = cnn_dir / "textcnn.pt"
        manifest_path = cnn_dir / "artifact_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("CNN artifact_manifest.json is missing.")

        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict[str, str] = json.load(f)

        for filename, expected_hash in manifest.items():
            filepath = cnn_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing CNN artifact: {filename}")
            actual_hash = calculate_file_sha256(filepath)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Hash mismatch for CNN artifact {filename}. "
                    f"Expected {expected_hash}, got {actual_hash}."
                )

        with open(cnn_dir / "vocab.json", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)

        with open(cnn_dir / "labels.json", encoding="utf-8") as f:
            self.labels: list[str] = json.load(f)

        with open(cnn_dir / "config.json", encoding="utf-8") as f:
            self.config: dict[str, object] = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=int(self.config["embedding_dim"]),
            num_filters=int(self.config["num_filters"]),
            kernel_sizes=list(self.config["kernel_sizes"]),  # type: ignore[arg-type]
            num_classes=int(self.config["num_classes"]),
            dropout=float(self.config["dropout"]),
        )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.threshold = load_threshold(CNN_THRESHOLD_PATH, weights_path)
        self.model_sha256 = calculate_file_sha256(weights_path)

    def predict(self, text: str) -> PredictionResult:
        token_ids = encode_text(text, self.vocab, max_length=int(self.config["max_length"]))
        x = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return format_prediction(self.labels, probabilities, self.threshold)


def get_predictor(backend: str = "auto") -> BaselinePredictor | CNNPredictor:
    if backend == "auto":
        backend = get_selected_backend()

    if backend == "baseline":
        return BaselinePredictor()

    if backend == "cnn":
        return CNNPredictor()

    raise ValueError(f"Unknown backend: {backend}")
