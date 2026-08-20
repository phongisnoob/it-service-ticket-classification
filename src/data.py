"""Dataset loading and deterministic split utilities.

The ``prepare_data`` DVC stage (src/prepare_data.py) is the **sole** producer
of split IDs and the data quality report.  ``load_data`` and ``split_data``
consume those persisted outputs so every downstream stage uses identical
partitions.

Canonical split design (70 / 10 / 10 / 10):
    train           — model fitting
    tune            — threshold selection (never used for training)
    calibration     — reserved for post-hoc calibration quality measurement
    test            — final held-out evaluation (never used before final report)

``split_data()`` returns (train, tune, test).  The calibration set is
available separately via ``load_calibration_split()``.

No fallback split generation: if persisted split files are absent, the
function raises ``FileNotFoundError`` with a clear instruction.  This prevents
silent use of a different split ratio (e.g. 70/15/15) than the canonical one.
"""

import hashlib
import json
import re

import pandas as pd

from src.hashing import calculate_file_sha256
from src.paths import DATA_PATH, REPORT_DATA_DIR


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def load_data(deduplicate: bool = True) -> pd.DataFrame:
    """Load and clean the ticket classification dataset.

    Uses a full SHA-256 ticket ID (not a 16-char prefix) so IDs are
    collision-resistant.  Blank normalized rows are rejected before
    deduplication.

    Side effects:
        When ``deduplicate=True`` and conflicting labels are found, a CSV
        report is written to ``reports/data/conflicting_duplicate_labels.csv``.
        This is intentional for developer visibility during local runs; the
        canonical conflict report is produced by ``prepare_data`` instead.
    """
    df = pd.read_csv(DATA_PATH)[["Document", "Topic_group"]].copy()
    df = df.dropna(subset=["Document", "Topic_group"]).copy()

    df["Document"] = df["Document"].astype(str)
    df["Topic_group"] = df["Topic_group"].astype(str)
    df["document_normalized"] = df["Document"].apply(normalize_text)

    # Reject blank normalized rows
    blank_mask = df["document_normalized"].str.strip() == ""
    if blank_mask.any():
        df = df[~blank_mask].copy()

    # Full SHA-256 row ID for collision resistance.
    df["ticket_id"] = df.apply(
        lambda row: hashlib.sha256(
            (row["document_normalized"] + "|" + row["Topic_group"]).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    if deduplicate:
        conflicts = df.groupby("document_normalized")["Topic_group"].nunique()
        conflicting_docs = conflicts[conflicts > 1].index

        if len(conflicting_docs) > 0:
            REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df[df["document_normalized"].isin(conflicting_docs)].to_csv(
                REPORT_DATA_DIR / "conflicting_duplicate_labels.csv"
            )
            df = df[~df["document_normalized"].isin(conflicting_docs)].copy()

        df = df.drop_duplicates(subset=["document_normalized"], keep="first").copy()

    result = df.set_index("ticket_id")
    assert result.index.is_unique, "ticket_id collisions after deduplication"
    return result


def validate_persisted_splits(
    df: pd.DataFrame,
    train_ids: "pd.Index",
    tune_ids: "pd.Index",
    test_ids: "pd.Index",
    manifest: dict[str, object],
    calib_ids: "pd.Index | None" = None,
) -> None:
    """Validate persisted splits against the current dataset.

    Checks:
    - Dataset SHA-256 matches manifest (detects modified CSV)
    - All split IDs exist in the current dataset
    - All splits are mutually disjoint (train/tune/test + optional calib)
    - Split sizes match manifest
    - For 3-way splits: train + tune + test covers the full dataset
    - For 4-way splits: calibration disjointness is also verified

    Args:
        df: The full cleaned dataset indexed by ticket_id.
        train_ids: IDs in the training split.
        tune_ids: IDs in the tune/validation split (used for threshold selection).
        test_ids: IDs in the held-out test split.
        manifest: Parsed data_manifest.json dict.
        calib_ids: Optional calibration split IDs. When provided, 4-way
            disjointness is verified.

    Raises:
        RuntimeError: On any integrity violation.
    """
    current_sha = calculate_file_sha256(DATA_PATH)
    stored_sha = manifest.get("dataset_sha256")

    if stored_sha is None:
        raise RuntimeError(
            "Persisted split manifest does not contain 'dataset_sha256'. "
            "Regenerate the splits by running: dvc repro prepare_data"
        )
    if current_sha != stored_sha:
        raise RuntimeError(
            "Dataset has changed since the persisted splits were created. "
            "Regenerate splits by running: dvc repro prepare_data"
        )

    train_set = set(train_ids)
    tune_set = set(tune_ids)
    test_set = set(test_ids)
    calib_set = set(calib_ids) if calib_ids is not None else set()
    current_ids = set(df.index)

    # Disjointness checks — always required
    pairs = [
        ("train", train_set, "tune", tune_set),
        ("train", train_set, "test", test_set),
        ("tune", tune_set, "test", test_set),
    ]
    if calib_ids is not None:
        pairs.extend([
            ("train", train_set, "calibration", calib_set),
            ("tune", tune_set, "calibration", calib_set),
            ("test", test_set, "calibration", calib_set),
        ])

    for name_a, set_a, name_b, set_b in pairs:
        overlap = set_a & set_b
        if overlap:
            raise RuntimeError(
                f"Persisted {name_a} and {name_b} splits overlap: "
                f"{len(overlap)} shared IDs."
            )

    # All persisted IDs must exist in current dataset
    all_persisted = train_set | tune_set | test_set | calib_set
    unknown = all_persisted - current_ids
    if unknown:
        raise RuntimeError(
            f"Persisted splits contain {len(unknown)} IDs not found in current dataset. "
            "Regenerate splits by running: dvc repro prepare_data"
        )

    # Size checks against manifest
    if manifest.get("train_rows") is not None and len(train_ids) != manifest["train_rows"]:
        raise RuntimeError("Persisted train split size does not match data_manifest.json.")
    if manifest.get("tune_rows") is not None and len(tune_ids) != manifest["tune_rows"]:
        raise RuntimeError("Persisted tune split size does not match data_manifest.json.")
    if manifest.get("test_rows") is not None and len(test_ids) != manifest["test_rows"]:
        raise RuntimeError("Persisted test split size does not match data_manifest.json.")
    if calib_ids is not None and manifest.get("calibration_rows") is not None:
        if len(calib_ids) != manifest["calibration_rows"]:
            raise RuntimeError(
                "Persisted calibration split size does not match data_manifest.json."
            )

    # Collective exhaustiveness — only enforceable when calibration split known
    has_calibration = calib_ids is not None
    if has_calibration:
        persisted_ids = train_set | tune_set | test_set | calib_set
    else:
        persisted_ids = train_set | tune_set | test_set

    if manifest.get("total_rows") is not None:
        total_manifest = int(str(manifest["total_rows"]))
        if len(persisted_ids) != total_manifest:
            # Only enforce strict equality for 3-way splits (calib absent)
            if not has_calibration:
                missing = current_ids - persisted_ids
                raise RuntimeError(
                    f"Persisted splits do not cover all dataset rows. "
                    f"Missing: {len(missing)} IDs. "
                    "Regenerate splits by running: dvc repro prepare_data"
                )


def load_calibration_split(df: pd.DataFrame) -> pd.DataFrame:
    """Load the calibration split from persisted IDs.

    The calibration set is the 10% reserved partition produced by
    ``prepare_data``. It is used for post-hoc calibration quality measurement
    (ECE, Brier score) **after** the model has been fitted and thresholded on
    separate data.  It must never be used for training or threshold selection.

    Raises:
        FileNotFoundError: If calibration_ids.csv is not present.
            Run ``dvc repro prepare_data`` to generate it.
    """
    calib_ids_path = REPORT_DATA_DIR / "calibration_ids.csv"
    if not calib_ids_path.exists():
        raise FileNotFoundError(
            f"Calibration split IDs not found: {calib_ids_path}. "
            "Run: dvc repro prepare_data"
        )
    calib_ids = pd.read_csv(calib_ids_path)["id"].values
    return df.loc[calib_ids].copy()


def split_data(
    df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, tune, test) DataFrames from persisted split IDs.

    The canonical 70/10/10/10 split (train/tune/calibration/test) is produced
    exclusively by the ``prepare_data`` DVC stage (src/prepare_data.py).
    This function loads and validates those persisted IDs.

    **No fallback split generation**: if the split files are absent, a
    ``FileNotFoundError`` is raised with a clear instruction. This prevents
    accidental use of a different split ratio.

    The calibration split (10%) is not returned here to keep the existing
    caller interface stable. Use ``load_calibration_split()`` to access it.

    Args:
        df: Pre-loaded dataset. If None, ``load_data()`` is called.
        random_state: Accepted for API compatibility; the split is always
            loaded from persisted files, so this parameter has no effect.

    Returns:
        (train_df, tune_df, test_df) — non-overlapping DataFrames.

    Raises:
        FileNotFoundError: If any required split file or manifest is absent.
        RuntimeError: If integrity validation fails (hash mismatch, overlap,
            size mismatch).
    """
    if df is None:
        df = load_data(deduplicate=True)

    train_ids_path = REPORT_DATA_DIR / "train_ids.csv"
    tune_ids_path = REPORT_DATA_DIR / "tune_ids.csv"
    test_ids_path = REPORT_DATA_DIR / "test_ids.csv"
    calib_ids_path = REPORT_DATA_DIR / "calibration_ids.csv"
    manifest_path = REPORT_DATA_DIR / "data_manifest.json"

    required = {
        "train_ids.csv": train_ids_path,
        "tune_ids.csv": tune_ids_path,
        "test_ids.csv": test_ids_path,
        "data_manifest.json": manifest_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Persisted split files missing: {missing}. "
            "Run: dvc repro prepare_data"
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, object] = json.load(f)

    train_ids = pd.read_csv(train_ids_path)["id"].values
    tune_ids = pd.read_csv(tune_ids_path)["id"].values
    test_ids = pd.read_csv(test_ids_path)["id"].values

    # Load calibration IDs if present (always expected from canonical pipeline)
    calib_ids: pd.Index | None = None
    if calib_ids_path.exists():
        calib_ids = pd.Index(pd.read_csv(calib_ids_path)["id"].values)

    validate_persisted_splits(df, pd.Index(train_ids), pd.Index(tune_ids),
                               pd.Index(test_ids), manifest, calib_ids)

    return df.loc[train_ids].copy(), df.loc[tune_ids].copy(), df.loc[test_ids].copy()
