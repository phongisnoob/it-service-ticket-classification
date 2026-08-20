import hashlib
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.hashing import calculate_file_sha256
from src.paths import DATA_PATH, REPORT_DATA_DIR


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping whitespace, and collapsing spaces.

    Args:
        text: Raw input text.

    Returns:
        Normalized text string.
    """
    return re.sub(r"\s+", " ", str(text).lower().strip())


def load_data(deduplicate: bool = True) -> pd.DataFrame:
    """Load and clean the ticket classification dataset."""
    df = pd.read_csv(DATA_PATH)[["Document", "Topic_group"]].copy()
    df = df.dropna(subset=["Document", "Topic_group"]).copy()

    df["Document"] = df["Document"].astype(str)
    df["Topic_group"] = df["Topic_group"].astype(str)
    df["document_normalized"] = df["Document"].apply(normalize_text)

    # Stable ticket ID for split reproducibility.
    df["ticket_id"] = df.apply(
        lambda row: hashlib.sha256(
            (row["document_normalized"] + "|" + row["Topic_group"]).encode("utf-8")
        ).hexdigest()[:16],
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

    return df.set_index("ticket_id")


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> None:
    """Persist split IDs and metadata required for reproducibility."""
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": train_df.index}).to_csv(REPORT_DATA_DIR / "train_ids.csv", index=False)
    pd.DataFrame({"id": val_df.index}).to_csv(REPORT_DATA_DIR / "val_ids.csv", index=False)
    pd.DataFrame({"id": test_df.index}).to_csv(REPORT_DATA_DIR / "test_ids.csv", index=False)

    manifest = {
        "dataset_sha256": calculate_file_sha256(DATA_PATH),
        "random_seed": random_state,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(train_df) + len(val_df) + len(test_df),
    }
    with open(REPORT_DATA_DIR / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


def _validate_split_manifest(manifest: dict[str, object]) -> None:
    """Validate that manifest contains required fields."""
    if manifest.get("dataset_sha256") is None:
        raise RuntimeError(
            "Persisted split manifest does not contain 'dataset_sha256'. Regenerate the splits."
        )


def _validate_split_ids(
    train_set: set[str],
    val_set: set[str],
    test_set: set[str],
    current_ids: set[str],
) -> None:
    """Validate that split IDs are valid and non-overlapping."""
    if not train_set.isdisjoint(val_set):
        raise RuntimeError("Persisted train and validation splits overlap.")
    if not train_set.isdisjoint(test_set):
        raise RuntimeError("Persisted train and test splits overlap.")
    if not val_set.isdisjoint(test_set):
        raise RuntimeError("Persisted validation and test splits overlap.")

    persisted_ids = train_set | val_set | test_set
    if persisted_ids != current_ids:
        missing = current_ids - persisted_ids
        unknown = persisted_ids - current_ids
        raise RuntimeError(
            f"Persisted splits do not exactly match the current dataset. "
            f"Missing from splits: {len(missing)}. Unknown persisted IDs: {len(unknown)}. "
            "Regenerate train/validation/test splits."
        )


def _validate_split_sizes(
    train_ids: "pd.Index",
    val_ids: "pd.Index",
    test_ids: "pd.Index",
    manifest: dict[str, object],
) -> None:
    """Validate that split sizes match manifest."""
    if manifest.get("train_rows") is not None and len(train_ids) != manifest["train_rows"]:
        raise RuntimeError("Persisted train split size does not match data_manifest.json.")
    if manifest.get("validation_rows") is not None and len(val_ids) != manifest["validation_rows"]:
        raise RuntimeError("Persisted validation split size does not match data_manifest.json.")
    if manifest.get("test_rows") is not None and len(test_ids) != manifest["test_rows"]:
        raise RuntimeError("Persisted test split size does not match data_manifest.json.")
    if (
        manifest.get("total_rows") is not None
        and len(train_ids) + len(val_ids) + len(test_ids) != manifest["total_rows"]
    ):
        raise RuntimeError("Persisted total split size does not match data_manifest.json.")


def validate_persisted_splits(
    df: pd.DataFrame,
    train_ids: "pd.Index",
    val_ids: "pd.Index",
    test_ids: "pd.Index",
    manifest: dict[str, object],
) -> None:
    """Validate persisted splits against the current dataset."""
    current_sha = calculate_file_sha256(DATA_PATH)
    stored_sha = manifest.get("dataset_sha256")

    if current_sha != stored_sha:
        raise RuntimeError(
            "Dataset has changed since the persisted splits were created. "
            "Regenerate train/validation/test splits."
        )

    _validate_split_manifest(manifest)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    current_ids = set(df.index)

    _validate_split_ids(train_set, val_set, test_set, current_ids)
    _validate_split_sizes(train_ids, val_ids, test_ids, manifest)


def split_data(
    df: pd.DataFrame | None = None,
    random_state: int = 42,
    use_persisted: bool = True,
    persist_manifest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames using a 70/15/15 stratified split.

    When persisted split files exist and the dataset SHA-256 still matches, the exact
    same split is reloaded. This ensures models are never evaluated on data they were
    trained on even if split() is called multiple times.
    """
    if df is None:
        df = load_data(deduplicate=True)

    train_ids_path = REPORT_DATA_DIR / "train_ids.csv"
    val_ids_path = REPORT_DATA_DIR / "val_ids.csv"
    test_ids_path = REPORT_DATA_DIR / "test_ids.csv"
    manifest_path = REPORT_DATA_DIR / "data_manifest.json"

    if use_persisted and all(
        p.exists() for p in [train_ids_path, val_ids_path, test_ids_path, manifest_path]
    ):
        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict[str, object] = json.load(f)

        train_ids = pd.read_csv(train_ids_path)["id"].values
        val_ids = pd.read_csv(val_ids_path)["id"].values
        test_ids = pd.read_csv(test_ids_path)["id"].values

        validate_persisted_splits(df, train_ids, val_ids, test_ids, manifest)

        return df.loc[train_ids].copy(), df.loc[val_ids].copy(), df.loc[test_ids].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["Topic_group"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Topic_group"],
        random_state=random_state,
    )

    if persist_manifest:
        save_split_manifest(train_df, val_df, test_df, random_state)

    return train_df, val_df, test_df
