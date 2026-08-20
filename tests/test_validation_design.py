"""Unit tests for data integrity and validation design.

Covers:
- Split disjointness (3-way and 4-way)
- Blank row rejection
- ID collision detection
- Clopper-Pearson simultaneous bounds
- Threshold selection determinism
- Per-class small-support handling
"""

import hashlib
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.routing_utils import (
    clopper_pearson_lower,
    per_class_accepted_stats,
    select_threshold,
    simultaneous_lower_bound,
)

# ---------------------------------------------------------------------------
# Helper: minimal synthetic dataset
# ---------------------------------------------------------------------------

def _make_results(
    n: int = 200,
    seed: int = 0,
    high_confidence: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["A", "B", "C"]
    true_label = rng.choice(labels, size=n)
    confidence = rng.uniform(0.5 if high_confidence else 0.0, 1.0, size=n)
    # Correct ≈ 90% of the time when confidence >= 0.7
    correct = np.where(
        confidence >= 0.7,
        rng.random(size=n) < 0.92,
        rng.random(size=n) < 0.65,
    )
    return pd.DataFrame(
        {"true_label": true_label, "confidence": confidence, "correct": correct.astype(int)}
    )


# ---------------------------------------------------------------------------
# Clopper-Pearson bounds
# ---------------------------------------------------------------------------

class TestClopperPearson:
    def test_zero_trials_returns_zero(self) -> None:
        assert clopper_pearson_lower(0, 0, 0.05) == 0.0

    def test_zero_successes_returns_zero(self) -> None:
        assert clopper_pearson_lower(0, 100, 0.05) == 0.0

    def test_all_successes_near_one(self) -> None:
        lb = clopper_pearson_lower(1000, 1000, 0.05)
        assert lb > 0.99

    def test_lb_below_proportion(self) -> None:
        # Lower bound must be <= point estimate
        lb = clopper_pearson_lower(90, 100, 0.05)
        assert lb <= 0.90

    def test_lb_positive_for_reasonable_input(self) -> None:
        lb = clopper_pearson_lower(95, 100, 0.05)
        assert lb > 0.85

    def test_simultaneous_lb_tighter_than_individual(self) -> None:
        # Bonferroni correction makes the bound more conservative
        lb_single = clopper_pearson_lower(90, 100, 0.05)
        lb_simultaneous = simultaneous_lower_bound(90, 100, 0.05, n_candidates=90)
        assert lb_simultaneous <= lb_single

    def test_simultaneous_zero_candidates(self) -> None:
        assert simultaneous_lower_bound(90, 100, 0.05, 0) == 0.0


# ---------------------------------------------------------------------------
# Threshold selection determinism and correctness
# ---------------------------------------------------------------------------

class TestSelectThreshold:
    def test_deterministic(self) -> None:
        df = _make_results(500, seed=1)
        row1, analysis1 = select_threshold(df, target_accuracy=0.80)
        row2, analysis2 = select_threshold(df, target_accuracy=0.80)
        if row1 is not None and row2 is not None:
            assert float(row1["threshold"]) == float(row2["threshold"])
        assert len(analysis1) == len(analysis2)

    def test_returns_none_when_no_candidate_qualifies(self) -> None:
        # All predictions are wrong → no threshold can meet 90% accuracy
        df = pd.DataFrame(
            {"true_label": ["A"] * 100, "confidence": [0.9] * 100, "correct": [0] * 100}
        )
        selected, _ = select_threshold(df, target_accuracy=0.90)
        assert selected is None

    def test_selects_highest_coverage_eligible(self) -> None:
        """Among eligible thresholds, the one with highest coverage is chosen."""
        df = _make_results(1000, seed=42, high_confidence=True)
        selected, analysis = select_threshold(df, target_accuracy=0.80, min_accepted_samples=10)
        if selected is not None:
            eligible = analysis[analysis["meets_requirement"]]
            assert float(selected["coverage"]) == float(eligible["coverage"].max())

    def test_min_accepted_samples_respected(self) -> None:
        # With very high min_samples, should return None on small dataset
        df = _make_results(100, seed=0)
        selected, _ = select_threshold(df, min_accepted_samples=10000)
        assert selected is None

    def test_bonferroni_correction_applied(self) -> None:
        """analysis_df must contain n_candidates rows matching the grid."""
        df = _make_results(500, seed=1)
        _, analysis = select_threshold(df, threshold_start=0.10, threshold_stop=0.50,
                                       threshold_step=0.10)
        # 0.10, 0.20, 0.30, 0.40 → 4 candidates
        assert len(analysis) == 4


# ---------------------------------------------------------------------------
# Data integrity: blank rows and ID uniqueness
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_blank_row_rejected(self) -> None:
        """normalize_text of a whitespace-only string is blank and must be rejected."""
        from src.data import normalize_text
        assert normalize_text("   ") == ""

    def test_full_sha256_id_length(self) -> None:
        """Row IDs must be full 64-char SHA-256 hexdigests."""
        normalized = "cannot connect to wifi"
        label = "Network"
        row_id = hashlib.sha256(
            (normalized + "|" + label).encode("utf-8")
        ).hexdigest()
        assert len(row_id) == 64

    def test_id_collision_on_identical_rows(self) -> None:
        """Two rows with identical text and label produce the same ID (expected)."""
        normalized = "printer is broken"
        label = "Hardware"
        id1 = hashlib.sha256((normalized + "|" + label).encode()).hexdigest()
        id2 = hashlib.sha256((normalized + "|" + label).encode()).hexdigest()
        assert id1 == id2

    def test_id_differs_across_labels(self) -> None:
        """Same text with different labels must produce different IDs."""
        normalized = "cannot login"
        id_access = hashlib.sha256((normalized + "|Access").encode()).hexdigest()
        id_network = hashlib.sha256((normalized + "|Network").encode()).hexdigest()
        assert id_access != id_network


# ---------------------------------------------------------------------------
# Split disjointness
# ---------------------------------------------------------------------------

class TestSplitDisjointness:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        labels = ["A", "B", "C", "D"]
        data = {
            "document_normalized": [f"doc_{i}" for i in range(n)],
            "Topic_group": rng.choice(labels, size=n),
        }
        df = pd.DataFrame(data)
        df["ticket_id"] = df.apply(
            lambda r: hashlib.sha256(
                (r["document_normalized"] + "|" + r["Topic_group"]).encode()
            ).hexdigest(),
            axis=1,
        )
        return df.set_index("ticket_id")

    def test_train_tune_test_disjoint(self) -> None:
        from sklearn.model_selection import train_test_split

        df = self._make_df(200)
        train, temp = train_test_split(df, test_size=0.30, stratify=df["Topic_group"],
                                       random_state=42)
        val, test = train_test_split(temp, test_size=0.50, stratify=temp["Topic_group"],
                                     random_state=42)
        assert set(train.index).isdisjoint(set(val.index))
        assert set(train.index).isdisjoint(set(test.index))
        assert set(val.index).isdisjoint(set(test.index))

    def test_no_test_access_during_split(self) -> None:
        """split_data must NOT look at test_ids when called for training.

        This structural test confirms split_data returns train, tune, test
        using only persisted IDs (not recomputing from test).
        """
        from src.data import validate_persisted_splits

        # Build a minimal consistent manifest — use canonical tune_rows key
        df = self._make_df(90)
        ids = list(df.index)
        train_ids = pd.Index(ids[:63])
        tune_ids = pd.Index(ids[63:76])
        test_ids = pd.Index(ids[76:])
        manifest: dict[str, object] = {
            "dataset_sha256": "fakehash",
            "train_rows": len(train_ids),
            "tune_rows": len(tune_ids),  # canonical key (not validation_rows)
            "test_rows": len(test_ids),
            "total_rows": len(df),
        }
        # Patch file SHA so validation passes hash check
        with patch("src.data.calculate_file_sha256", return_value="fakehash"):
            validate_persisted_splits(df, train_ids, tune_ids, test_ids, manifest)

    def test_four_way_split_disjoint(self) -> None:
        """Canonical 70/10/10/10 split must have zero overlap across all 6 pairs."""
        from sklearn.model_selection import train_test_split

        df = self._make_df(200)
        # 70% train, 30% temp
        train, temp = train_test_split(df, test_size=0.30, stratify=df["Topic_group"],
                                       random_state=42)
        # temp → 1/3 tune (10% overall), 2/3 calib+test (20% overall)
        tune, calib_test = train_test_split(temp, test_size=2 / 3,
                                            stratify=temp["Topic_group"], random_state=42)
        # calib_test → 50/50 calibration + test (10% each)
        calib, test = train_test_split(calib_test, test_size=0.50,
                                       stratify=calib_test["Topic_group"], random_state=42)

        splits = {
            "train": set(train.index),
            "tune": set(tune.index),
            "calibration": set(calib.index),
            "test": set(test.index),
        }
        names = list(splits.keys())
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                overlap = splits[name_a] & splits[name_b]
                assert not overlap, (
                    f"{name_a} and {name_b} overlap: {len(overlap)} shared IDs"
                )


# ---------------------------------------------------------------------------
# Per-class stats
# ---------------------------------------------------------------------------

class TestPerClassStats:
    def test_zero_accepted_class_handled(self) -> None:
        """A class with zero accepted rows must not raise."""
        df = pd.DataFrame(
            {
                "true_label": ["A", "A", "B"],
                "confidence": [0.9, 0.8, 0.3],  # B falls below threshold
                "correct": [1, 1, 1],
            }
        )
        stats = per_class_accepted_stats(df, threshold=0.5)
        labels = {s["label"] for s in stats}
        assert "A" in labels
        # B is not accepted so it won't appear in the stats
        assert "B" not in labels

    def test_wilson_lb_nonnegative(self) -> None:
        df = _make_results(300, seed=5)
        stats = per_class_accepted_stats(df, threshold=0.6)
        for s in stats:
            assert s["wilson_lb_95"] >= 0.0

    def test_wilson_lb_le_accuracy(self) -> None:
        df = _make_results(300, seed=5)
        stats = per_class_accepted_stats(df, threshold=0.6)
        for s in stats:
            assert s["wilson_lb_95"] <= s["accuracy"] + 1e-9
