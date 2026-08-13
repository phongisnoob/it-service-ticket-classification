import hashlib
import json
import re
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.hashing import calculate_file_sha256
from src.paths import DATA_PATH, REPORT_DATA_DIR


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping, and collapsing whitespace."""
    return re.sub(
        r"\s+",
        " ",
        str(text).lower().strip(),
    )


def load_data(
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load and clean the ticket classification dataset."""
    df = pd.read_csv(DATA_PATH)

    # Keep only columns required for modeling.
    df = df[
        [
            "Document",
            "Topic_group",
        ]
    ].copy()

    # Remove rows missing either text or label.
    df = df.dropna(
        subset=[
            "Document",
            "Topic_group",
        ]
    ).copy()

    # Ensure stable string representation.
    df["Document"] = df["Document"].astype(str)
    df["Topic_group"] = df["Topic_group"].astype(str)

    # Normalize text for duplicate detection.
    df["document_normalized"] = df["Document"].apply(normalize_text)

    # Create a stable ticket ID based on normalized text + label.
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
            conflicting_df = df[df["document_normalized"].isin(conflicting_docs)]
            REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
            conflicting_df.to_csv(REPORT_DATA_DIR / "conflicting_duplicate_labels.csv")

            df = df[~df["document_normalized"].isin(conflicting_docs)].copy()

        df = df.drop_duplicates(
            subset=["document_normalized"],
            keep="first",
        ).copy()

    df = df.set_index("ticket_id")

    return df


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> None:
    """Persist split IDs and metadata required for reproducibility."""
    REPORT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ids_path = REPORT_DATA_DIR / "train_ids.csv"
    val_ids_path = REPORT_DATA_DIR / "val_ids.csv"
    test_ids_path = REPORT_DATA_DIR / "test_ids.csv"
    manifest_path = REPORT_DATA_DIR / "data_manifest.json"

    pd.DataFrame({"id": train_df.index}).to_csv(
        train_ids_path,
        index=False,
    )

    pd.DataFrame({"id": val_df.index}).to_csv(
        val_ids_path,
        index=False,
    )

    pd.DataFrame({"id": test_df.index}).to_csv(
        test_ids_path,
        index=False,
    )

    manifest = {
        "dataset_sha256": calculate_file_sha256(DATA_PATH),
        "random_seed": random_state,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": (len(train_df) + len(val_df) + len(test_df)),
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )


def validate_persisted_splits(
    df: pd.DataFrame,
    train_ids: Any,
    val_ids: Any,
    test_ids: Any,
    manifest: dict,  # type: ignore
) -> None:
    """Validate persisted splits against the current dataset."""
    current_dataset_sha256 = calculate_file_sha256(DATA_PATH)

    stored_dataset_sha256 = manifest.get("dataset_sha256")

    if stored_dataset_sha256 is None:
        raise RuntimeError(
            "Persisted split manifest does not contain 'dataset_sha256'. Regenerate the splits."
        )

    if current_dataset_sha256 != stored_dataset_sha256:
        raise RuntimeError(
            "Dataset has changed since the persisted "
            "splits were created. Regenerate "
            "train/validation/test splits."
        )

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)

    current_ids = set(df.index)

    # Ensure train, validation, and test are mutually exclusive.
    if not train_id_set.isdisjoint(val_id_set):
        raise RuntimeError("Persisted train and validation splits overlap.")

    if not train_id_set.isdisjoint(test_id_set):
        raise RuntimeError("Persisted train and test splits overlap.")

    if not val_id_set.isdisjoint(test_id_set):
        raise RuntimeError("Persisted validation and test splits overlap.")

    persisted_ids = train_id_set | val_id_set | test_id_set

    # Every row in the current dataset must belong to exactly one split.
    if persisted_ids != current_ids:
        missing_from_splits = current_ids - persisted_ids
        unknown_in_splits = persisted_ids - current_ids

        raise RuntimeError(
            "Persisted splits do not exactly match "
            "the current dataset. "
            f"Missing from splits: "
            f"{len(missing_from_splits)}. "
            f"Unknown persisted IDs: "
            f"{len(unknown_in_splits)}. "
            "Regenerate train/validation/test splits."
        )

    # Validate counts stored in the manifest.
    expected_train_rows = manifest.get("train_rows")
    expected_val_rows = manifest.get("validation_rows")
    expected_test_rows = manifest.get("test_rows")
    expected_total_rows = manifest.get("total_rows")

    if expected_train_rows is not None and len(train_ids) != expected_train_rows:
        raise RuntimeError("Persisted train split size does not match data_manifest.json.")

    if expected_val_rows is not None and len(val_ids) != expected_val_rows:
        raise RuntimeError("Persisted validation split size does not match data_manifest.json.")

    if expected_test_rows is not None and len(test_ids) != expected_test_rows:
        raise RuntimeError("Persisted test split size does not match data_manifest.json.")

    total_rows = len(train_ids) + len(val_ids) + len(test_ids)

    if expected_total_rows is not None and total_rows != expected_total_rows:
        raise RuntimeError("Persisted total split size does not match data_manifest.json.")


def split_data(
    df: pd.DataFrame | None = None,
    random_state: int = 42,
    use_persisted: bool = True,
    persist_manifest: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Split the dataset into train, validation, and test sets.

    When persisted split files exist, they are reused only if:

    - the raw dataset SHA-256 matches the saved manifest;
    - train/validation/test IDs do not overlap;
    - every current dataset row belongs to exactly one split;
    - saved split counts match the manifest.

    Otherwise, a new stratified 70/15/15 split is created.
    """
    if df is None:
        df = load_data(deduplicate=True)

    train_ids_path = REPORT_DATA_DIR / "train_ids.csv"
    val_ids_path = REPORT_DATA_DIR / "val_ids.csv"
    test_ids_path = REPORT_DATA_DIR / "test_ids.csv"
    manifest_path = REPORT_DATA_DIR / "data_manifest.json"

    persisted_files_exist = all(
        [
            train_ids_path.exists(),
            val_ids_path.exists(),
            test_ids_path.exists(),
            manifest_path.exists(),
        ]
    )

    if use_persisted and persisted_files_exist:
        with open(
            manifest_path,
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        train_ids = pd.read_csv(train_ids_path)["id"].values

        val_ids = pd.read_csv(val_ids_path)["id"].values

        test_ids = pd.read_csv(test_ids_path)["id"].values

        validate_persisted_splits(
            df=df,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            manifest=manifest,
        )

        train_df = df.loc[train_ids].copy()

        val_df = df.loc[val_ids].copy()

        test_df = df.loc[test_ids].copy()

        return (
            train_df,
            val_df,
            test_df,
        )

    # -------------------------------------------------
    # Create new stratified 70 / 15 / 15 split.
    # -------------------------------------------------

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
        save_split_manifest(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            random_state=random_state,
        )

    return (
        train_df,
        val_df,
        test_df,
    )
