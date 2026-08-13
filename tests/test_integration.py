from typing import Any

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
    calculate_sha256,
    get_predictor,
    load_threshold,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_MODEL_PATH = ROOT_DIR / "artifacts" / "baseline.joblib"
BASELINE_THRESHOLD_PATH_LOCAL = (
    ROOT_DIR / "reports" / "metrics" / "baseline_selected_threshold.json"
)


def _model_and_threshold_compatible() -> Any:
    """Check if the model and threshold SHA-256 hashes match."""
    if not BASELINE_MODEL_PATH.exists() or not BASELINE_THRESHOLD_PATH_LOCAL.exists():
        return False
    try:
        with open(BASELINE_THRESHOLD_PATH_LOCAL, "r") as f:
            config = json.load(f)
        expected = config.get("model_sha256", "")
        actual = calculate_sha256(BASELINE_MODEL_PATH)
        return expected == actual
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


@requires_predictor
class TestBaselineInference:
    """Test the real baseline model pipeline end-to-end."""

    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls) -> Any:
        return BaselinePredictor()

    def test_predict_returns_valid_structure(self, predictor: Any) -> None:
        result = predictor.predict("I forgot my password and cannot access my account")

        assert result["category"]
        assert isinstance(result["category"], str)
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["threshold"] <= 1
        assert isinstance(result["needs_manual_review"], bool)
        assert len(result["top_3"]) == 3

    def test_top3_sorted_descending(self, predictor: Any) -> None:
        result = predictor.predict("My laptop keyboard is not working")

        probabilities = [item["probability"] for item in result["top_3"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_routing_flag_consistent(self, predictor: Any) -> None:
        result = predictor.predict("I need administrator access to install software")

        expected_review = result["confidence"] < result["threshold"]
        assert result["needs_manual_review"] == expected_review

    def test_probabilities_are_valid(self, predictor: Any) -> None:
        result = predictor.predict("Please upgrade my storage allocation")

        for item in result["top_3"]:
            assert 0 <= item["probability"] <= 1
            assert isinstance(item["category"], str)

    def test_model_sha256_is_set(self, predictor: Any) -> None:
        assert predictor.model_sha256 is not None
        assert len(predictor.model_sha256) == 64  # SHA-256 hex length


@requires_artifacts
class TestSHAValidation:
    """Test model/threshold SHA-256 integrity checks."""

    def test_sha_mismatch_raises(self, tmp_path: Any) -> None:
        """Verify RuntimeError when model SHA doesn't match threshold config."""
        fake_threshold = {
            "threshold": 0.40,
            "model_sha256": "0" * 64,
        }

        threshold_path = tmp_path / "fake_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(fake_threshold, f)

        with pytest.raises(RuntimeError, match="different model versions"):
            load_threshold(threshold_path, BASELINE_MODEL_PATH)

    def test_missing_sha_in_config_raises(self, tmp_path: Any) -> None:
        """Verify RuntimeError when threshold config has no SHA."""
        fake_threshold = {"threshold": 0.40}

        threshold_path = tmp_path / "no_sha_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(fake_threshold, f)

        with pytest.raises(RuntimeError, match="does not contain"):
            load_threshold(threshold_path, BASELINE_MODEL_PATH)

    def test_missing_model_raises(self, tmp_path: Any) -> None:
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
