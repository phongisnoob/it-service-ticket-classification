import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference import BaselinePredictor, calculate_sha256


def test_ml_smoke():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Create synthetic data
        texts = [
            "password reset request",
            "forgot my password",
            "need to change password",
            "network is down",
            "cannot connect to wifi",
            "internet not working",
            "printer is out of paper",
            "need new toner for printer",
            "printer jam",
        ]
        y = [
            "Access",
            "Access",
            "Access",
            "Network",
            "Network",
            "Network",
            "Hardware",
            "Hardware",
            "Hardware",
        ]

        # 2. Train a real TF-IDF + LogisticRegression pipeline
        pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())])
        pipeline.fit(texts, y)

        # 3. Serialize it with joblib
        model_path = tmp_path / "baseline.joblib"
        joblib.dump(pipeline, model_path)

        # 4. Create a threshold artifact containing the model SHA256
        model_sha256 = calculate_sha256(model_path)
        threshold_config = {"threshold": 0.85, "model_sha256": model_sha256}
        threshold_path = tmp_path / "baseline_selected_threshold.json"
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_config, f)

        # 5. Load the model through the production BaselinePredictor using mocks for paths
        with (
            patch("src.inference.ARTIFACT_DIR", tmp_path),
            patch("src.inference.BASELINE_THRESHOLD_PATH", threshold_path),
        ):
            predictor = BaselinePredictor()

            # 6. Make a real prediction
            prediction = predictor.predict("I need to reset my password")

            # 7. Validate confidence, threshold, output structure, and artifact hash handling
            assert "category" in prediction
            assert prediction["category"] in ["Access", "Network", "Hardware"]
            assert "confidence" in prediction
            assert 0.0 <= prediction["confidence"] <= 1.0
            assert prediction["threshold"] == 0.85
            assert prediction["needs_manual_review"] == (prediction["confidence"] < 0.85)
            assert "top_3" in prediction
            assert len(prediction["top_3"]) == 3
            assert prediction["top_3"][0]["category"] == prediction["category"]
