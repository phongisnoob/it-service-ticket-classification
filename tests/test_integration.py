"""Integration tests for the real ML serving stack.

These tests load actual model artifacts and verify that the full
inference pipeline works end-to-end. They require trained model
files to be present (not run in CI).
"""

import json
from pathlib import Path

import pytest

from src.inference import (
    BaselinePredictor,
    CNNPredictor,
    calculate_sha256,
    get_predictor,
    load_threshold,
)
from src.paths import BASELINE_MODEL_PATH, ROOT_DIR

BASELINE_THRESHOLD_PATH_LOCAL = ROOT_DIR / "reports" / "metrics" / "baseline_selected_threshold.json"
CNN_WEIGHTS_PATH = ROOT_DIR / "artifacts" / "cnn" / "textcnn.pt"
CNN_THRESHOLD_PATH_LOCAL = ROOT_DIR / "reports" / "metrics" / "selected_threshold.json"
CNN_MANIFEST_PATH = ROOT_DIR / "artifacts" / "cnn" / "artifact_manifest.json"


def _model_and_threshold_compatible() -> bool:
    """Check if the model and threshold SHA-256 hashes match."""
    if not BASELINE_MODEL_PATH.exists() or not BASELINE_THRESHOLD_PATH_LOCAL.exists():
        return False
    try:
        with open(BASELINE_THRESHOLD_PATH_LOCAL) as f:
            config = json.load(f)
        expected: str = config.get("model_sha256", "")
        actual = calculate_sha256(BASELINE_MODEL_PATH)
        return expected == actual
    except Exception:
        return False


def _cnn_and_threshold_compatible() -> bool:
    """Check if torch is installed and CNN model, manifest, and threshold SHA-256 hashes match."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    if not CNN_WEIGHTS_PATH.exists() or not CNN_THRESHOLD_PATH_LOCAL.exists() or not CNN_MANIFEST_PATH.exists():
        return False
    try:
        with open(CNN_THRESHOLD_PATH_LOCAL) as f:
            config = json.load(f)
        expected: str = config.get("model_sha256", "")
        actual = calculate_sha256(CNN_WEIGHTS_PATH)
        if expected != actual:
            return False
        with open(CNN_MANIFEST_PATH) as f:
            manifest: dict[str, str] = json.load(f)
        for fname, exp_hash in manifest.items():
            fpath = ROOT_DIR / "artifacts" / "cnn" / fname
            if not fpath.exists() or calculate_sha256(fpath) != exp_hash:
                return False
        return True
    except Exception:
        return False


# Tests that need model artifacts on disk
requires_artifacts = pytest.mark.skipif(
    not BASELINE_MODEL_PATH.exists() or not BASELINE_THRESHOLD_PATH_LOCAL.exists(),
    reason="Model artifacts not found — run training first",
)

# Tests that need a working predictor (model + matching threshold)
requires_predictor = pytest.mark.skipif(
    not _model_and_threshold_compatible(),
    reason="Model/threshold SHA mismatch — run threshold analysis first",
)

requires_cnn_predictor = pytest.mark.skipif(
    not _cnn_and_threshold_compatible(),
    reason="CNN artifacts/manifest/threshold missing or SHA mismatch — run CNN training first",
)


@requires_predictor
class TestBaselineInference:
    """Test the real baseline model pipeline end-to-end."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls) -> BaselinePredictor:
        return BaselinePredictor()

    def test_predict_returns_valid_structure(self, predictor: BaselinePredictor) -> None:
        result = predictor.predict("I forgot my password and cannot access my account")

        assert result["category"]
        assert isinstance(result["category"], str)
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["threshold"] <= 1
        assert isinstance(result["needs_manual_review"], bool)
        assert len(result["top_3"]) == 3

    def test_top3_sorted_descending(self, predictor: BaselinePredictor) -> None:
        result = predictor.predict("My laptop keyboard is not working")

        probabilities = [item["probability"] for item in result["top_3"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_routing_flag_consistent(self, predictor: BaselinePredictor) -> None:
        result = predictor.predict("I need administrator access to install software")

        expected_review = result["confidence"] < result["threshold"]
        assert result["needs_manual_review"] == expected_review

    def test_probabilities_are_valid(self, predictor: BaselinePredictor) -> None:
        result = predictor.predict("Please upgrade my storage allocation")

        for item in result["top_3"]:
            assert 0 <= item["probability"] <= 1
            assert isinstance(item["category"], str)

    def test_model_sha256_is_set(self, predictor: BaselinePredictor) -> None:
        assert predictor.model_sha256 is not None
        assert len(predictor.model_sha256) == 64  # SHA-256 hex digest length


@requires_cnn_predictor
class TestCNNInference:
    """Test the real CNN model pipeline end-to-end."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls) -> CNNPredictor:
        return CNNPredictor()

    def test_predict_returns_valid_structure(self, predictor: CNNPredictor) -> None:
        result = predictor.predict("I cannot access the VPN from home")

        assert result["category"]
        assert isinstance(result["category"], str)
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["threshold"] <= 1
        assert isinstance(result["needs_manual_review"], bool)
        assert len(result["top_3"]) == 3

    def test_top3_sorted_descending(self, predictor: CNNPredictor) -> None:
        result = predictor.predict("Printer not working in office 3B")

        probabilities = [item["probability"] for item in result["top_3"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_model_sha256_is_set(self, predictor: CNNPredictor) -> None:
        assert predictor.model_sha256 is not None
        assert len(predictor.model_sha256) == 64


@requires_artifacts
class TestSHAValidation:
    """Test model/threshold SHA-256 integrity checks."""

    def test_sha_mismatch_raises(self, tmp_path: Path) -> None:
        """Verify RuntimeError when model SHA doesn't match threshold config."""
        fake_threshold = {"threshold": 0.40, "model_sha256": "0" * 64}
        threshold_path = tmp_path / "fake_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(fake_threshold, f)

        with pytest.raises(RuntimeError, match="different model versions"):
            load_threshold(threshold_path, BASELINE_MODEL_PATH)

    def test_missing_sha_in_config_raises(self, tmp_path: Path) -> None:
        """Verify RuntimeError when threshold config has no SHA."""
        fake_threshold = {"threshold": 0.40}
        threshold_path = tmp_path / "no_sha_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(fake_threshold, f)

        with pytest.raises(RuntimeError, match="does not contain"):
            load_threshold(threshold_path, BASELINE_MODEL_PATH)

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError when model file is absent."""
        threshold_path = tmp_path / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": 0.40, "model_sha256": "abc"}, f)

        missing_model = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_threshold(threshold_path, missing_model)


class TestPredictorFactory:
    """Test get_predictor factory function."""

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_predictor("nonexistent")

    @requires_predictor
    def test_baseline_backend(self) -> None:
        predictor = get_predictor("baseline")
        assert isinstance(predictor, BaselinePredictor)

    @requires_cnn_predictor
    def test_cnn_backend(self) -> None:
        predictor = get_predictor("cnn")
        assert isinstance(predictor, CNNPredictor)
