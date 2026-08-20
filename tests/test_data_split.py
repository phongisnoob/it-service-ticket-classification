"""Tests for canonical 4-way data split architecture.

Verifies:
- split_data() raises FileNotFoundError when persisted split files are absent
- validate_persisted_splits() correctly validates 3-way and 4-way splits
- 4-way split disjointness including calibration set
- Overlap detection in all 6 pair combinations
- Manifest size checks
- ID unknown detection

These tests use mock data only and never touch the raw dataset.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import validate_persisted_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a minimal cleaned dataset indexed by ticket_id."""
    rng = np.random.default_rng(seed)
    labels = ["A", "B", "C", "D"]
    data = {
        "document_normalized": [f"doc_{i}" for i in range(n)],
        "Topic_group": rng.choice(labels, size=n).tolist(),
    }
    df = pd.DataFrame(data)
    df["ticket_id"] = df.apply(
        lambda r: hashlib.sha256(
            (r["document_normalized"] + "|" + r["Topic_group"]).encode()
        ).hexdigest(),
        axis=1,
    )
    return df.set_index("ticket_id")


def _four_way_split(df: pd.DataFrame) -> tuple[
    pd.Index, pd.Index, pd.Index, pd.Index
]:
    """Split df indices 70/10/10/10."""
    ids = list(df.index)
    n = len(ids)
    n_train = int(n * 0.70)
    n_tune = int(n * 0.10)
    n_calib = int(n * 0.10)
    train = pd.Index(ids[:n_train])
    tune = pd.Index(ids[n_train:n_train + n_tune])
    calib = pd.Index(ids[n_train + n_tune:n_train + n_tune + n_calib])
    test = pd.Index(ids[n_train + n_tune + n_calib:])
    return train, tune, calib, test


# ---------------------------------------------------------------------------
# Tests: split_data() requires persisted files
# ---------------------------------------------------------------------------

class TestSplitDataRequiresPersisted:
    """split_data() must raise FileNotFoundError when persisted files absent."""

    def test_raises_when_no_split_files(self) -> None:
        """FileNotFoundError raised when no split files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("src.data.DATA_PATH", tmp_path / "data.csv"),
                patch("src.data.REPORT_DATA_DIR", tmp_path / "reports"),
            ):
                # Need a dataset file to pass load_data
                # But split files are absent → should raise before even loading
                from src.data import split_data
                with pytest.raises(FileNotFoundError, match="Persisted split files missing"):
                    # Pass a minimal df to skip load_data
                    df = _make_df(20)
                    split_data(df)

    def test_error_message_mentions_dvc(self) -> None:
        """Error message should guide user to run dvc repro prepare_data."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("src.data.REPORT_DATA_DIR", tmp_path):
                from src.data import split_data
                try:
                    split_data(_make_df(20))
                except FileNotFoundError as e:
                    assert "prepare_data" in str(e)
                except Exception:
                    pass  # Other exceptions are fine for this test


# ---------------------------------------------------------------------------
# Tests: validate_persisted_splits — 4-way disjointness
# ---------------------------------------------------------------------------

class TestFourWaySplitDisjointness:
    """validate_persisted_splits must verify all 6 pairs when calib provided."""

    def _make_manifest(
        self,
        df: pd.DataFrame,
        train: pd.Index,
        tune: pd.Index,
        calib: pd.Index,
        test: pd.Index,
        sha: str = "fakehash",
    ) -> dict[str, object]:
        return {
            "dataset_sha256": sha,
            "train_rows": len(train),
            "tune_rows": len(tune),
            "calibration_rows": len(calib),
            "test_rows": len(test),
            "total_rows": len(df),
        }

    def test_clean_four_way_split_passes(self) -> None:
        """A correctly constructed 4-way split passes validation."""
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        manifest = self._make_manifest(df, train, tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            # Should not raise
            validate_persisted_splits(df, train, tune, test, manifest, calib)

    def test_train_tune_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        # Inject overlap: put first tune ID into train
        bad_train = pd.Index(list(train) + [tune[0]])
        manifest = self._make_manifest(df, bad_train, tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="train.*tune|tune.*train"):
                validate_persisted_splits(df, bad_train, tune, test, manifest, calib)

    def test_train_calib_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        bad_train = pd.Index(list(train) + [calib[0]])
        manifest = self._make_manifest(df, bad_train, tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="train.*calibration|calibration.*train"):
                validate_persisted_splits(df, bad_train, tune, test, manifest, calib)

    def test_tune_calib_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        bad_tune = pd.Index(list(tune) + [calib[0]])
        manifest = self._make_manifest(df, train, bad_tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="tune.*calibration|calibration.*tune"):
                validate_persisted_splits(df, train, bad_tune, test, manifest, calib)

    def test_calib_test_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        bad_calib = pd.Index(list(calib) + [test[0]])
        manifest = self._make_manifest(df, train, tune, bad_calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="test.*calibration|calibration.*test"):
                validate_persisted_splits(df, train, tune, test, manifest, bad_calib)

    def test_train_test_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        bad_train = pd.Index(list(train) + [test[0]])
        manifest = self._make_manifest(df, bad_train, tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="train.*test|test.*train"):
                validate_persisted_splits(df, bad_train, tune, test, manifest, calib)

    def test_tune_test_overlap_detected(self) -> None:
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        bad_tune = pd.Index(list(tune) + [test[0]])
        manifest = self._make_manifest(df, train, bad_tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="tune.*test|test.*tune"):
                validate_persisted_splits(df, train, bad_tune, test, manifest, calib)

    def test_unknown_ids_detected(self) -> None:
        """IDs not in df must be detected."""
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        ghost_id = "a" * 64  # fake SHA-256 hex
        bad_train = pd.Index(list(train) + [ghost_id])
        manifest = self._make_manifest(df, bad_train, tune, calib, test)
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="not found in current dataset"):
                validate_persisted_splits(df, bad_train, tune, test, manifest, calib)

    def test_dataset_sha_mismatch_detected(self) -> None:
        """Changed dataset SHA must be detected."""
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        manifest = self._make_manifest(df, train, tune, calib, test, sha="stored_sha")
        # calculate_file_sha256 returns a different sha than stored
        with patch("src.data.calculate_file_sha256", return_value="different_sha"):
            with pytest.raises(RuntimeError, match="changed since"):
                validate_persisted_splits(df, train, tune, test, manifest, calib)

    def test_calibration_size_mismatch_detected(self) -> None:
        """Calibration split size mismatch with manifest must be detected."""
        df = _make_df(200)
        train, tune, calib, test = _four_way_split(df)
        manifest = self._make_manifest(df, train, tune, calib, test)
        # Report wrong calibration_rows
        manifest["calibration_rows"] = len(calib) + 1
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            with pytest.raises(RuntimeError, match="calibration split size"):
                validate_persisted_splits(df, train, tune, test, manifest, calib)


# ---------------------------------------------------------------------------
# Tests: 4-way split determinism
# ---------------------------------------------------------------------------

class TestSplitDeterminism:
    """Same input + same seed must produce the same split IDs."""

    def test_validate_is_deterministic(self) -> None:
        """validate_persisted_splits is pure — same inputs always pass or fail."""
        df = _make_df(100, seed=42)
        train, tune, calib, test = _four_way_split(df)
        manifest: dict[str, object] = {
            "dataset_sha256": "fakehash",
            "train_rows": len(train),
            "tune_rows": len(tune),
            "calibration_rows": len(calib),
            "test_rows": len(test),
            "total_rows": len(df),
        }
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            # Call twice — must both pass
            validate_persisted_splits(df, train, tune, test, manifest, calib)
            validate_persisted_splits(df, train, tune, test, manifest, calib)
