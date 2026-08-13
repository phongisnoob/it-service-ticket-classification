import json
from pathlib import Path

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

THRESHOLD_PATH = (
    ROOT_DIR
    / "reports"
    / "metrics"
    / "selected_threshold.json"
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

def load_cnn_threshold():

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    return float(
        config["threshold"]
    )

def load_threshold(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    return float(
        config["threshold"]
    )

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

        self.model = joblib.load(
            model_path
        )

        self.labels = (
            self.model
            .named_steps["classifier"]
            .classes_
        )

        self.threshold = load_threshold(
            BASELINE_THRESHOLD_PATH
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
            cnn_dir / "textcnn.pt",
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
            CNN_THRESHOLD_PATH
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