import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import torch

from src.cnn_data import encode_text
from src.textcnn import TextCNN


ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = (
    ROOT_DIR
    / "artifacts"
)


BASELINE_THRESHOLD_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "baseline_selected_threshold.json"
)

CNN_THRESHOLD_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "selected_threshold.json"
)


def calculate_sha256(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_threshold(
    threshold_path,
    model_path,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    with open(
        threshold_path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    expected_hash = config.get(
        "model_sha256"
    )

    if not expected_hash:
        raise RuntimeError(
            "Threshold config does not contain "
            "a model SHA-256 hash. "
            "Run threshold selection again."
        )

    actual_hash = calculate_sha256(
        model_path
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "The model and threshold were generated "
            "from different model versions. "
            "Run threshold selection again."
        )

    threshold = float(
        config["threshold"]
    )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return threshold

def format_prediction(
    labels,
    probabilities,
    threshold,
):

    order = np.argsort(
        probabilities
    )[::-1]

    best_index = int(
        order[0]
    )

    confidence = float(
        probabilities[best_index]
    )

    top_3 = [
        {
            "category":
                str(labels[index]),

            "probability":
                float(
                    probabilities[index]
                ),
        }

        for index in order[:3]
    ]

    return {
        "category":
            str(
                labels[best_index]
            ),
        "confidence":
            confidence,
        "threshold": 
            float(threshold),
        "needs_manual_review":
            confidence < threshold,
        "top_3":
            top_3,
    }


# Logistic Regression
    

class BaselinePredictor:

    def __init__(self):

        model_path = (
            ARTIFACT_DIR
            / "baseline.joblib"
        )

        self.threshold = load_threshold(
            BASELINE_THRESHOLD_PATH,
            model_path,
        )

        self.model = joblib.load(
            model_path
        )

        self.labels = (
            self.model
            .named_steps["classifier"]
            .classes_
        )


    def predict(
        self,
        text,
    ):

        probabilities = (
            self.model
            .predict_proba([text])[0]
        )

        return format_prediction(
            self.labels,
            probabilities,
            self.threshold,
        )

# CNN


class CNNPredictor:

    def __init__(self):
        
        cnn_dir = (
            ARTIFACT_DIR
            / "cnn"
        )

        weights_path = (
            cnn_dir
            / "textcnn.pt"
        )
        with open(
            cnn_dir / "vocab.json",
            encoding="utf-8",
        ) as f:

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


        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        self.model = TextCNN(
            vocab_size=len(
                self.vocab
            ),

            embedding_dim=
                self.config[
                    "embedding_dim"
                ],

            num_filters=
                self.config[
                    "num_filters"
                ],

            kernel_sizes=
                self.config[
                    "kernel_sizes"
                ],

            num_classes=
                self.config[
                    "num_classes"
                ],

            dropout=
                self.config[
                    "dropout"
                ],
        )


        state_dict = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=True,
        )


        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.threshold = load_threshold(
            CNN_THRESHOLD_PATH,
            weights_path,
        )


    def predict(
        self,
        text,
    ):
        
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
    backend="baseline",
):

    if backend == "baseline":

        return BaselinePredictor()


    if backend == "cnn":

        return CNNPredictor()


    raise ValueError(
        f"Unknown backend: {backend}"
    )