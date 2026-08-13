import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "all_tickets_processed_improved_v3.csv"
)

REPORT_DATA_DIR = (
    ROOT_DIR
    / "reports"
    / "data"
)


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase, stripping, and collapsing whitespace."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def calculate_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_data(deduplicate: bool = True):
    """Load raw dataset, drop missing rows, and optionally deduplicate normalized documents."""
    df = pd.read_csv(DATA_PATH)

    df = df[["Document", "Topic_group"]].copy()

    df = df.dropna()

    df["Document"] = df["Document"].astype(str)
    df["Topic_group"] = df["Topic_group"].astype(str)
    df["document_normalized"] = df["Document"].apply(normalize_text)

    if deduplicate:
        df = df.drop_duplicates(
            subset=["document_normalized"],
            keep="first",
        ).copy()

    return df


def save_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
):
    """Persist split indices and metadata manifest."""
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"id": train_df.index}).to_csv(
        REPORT_DATA_DIR / "train_ids.csv",
        index=False,
    )
    pd.DataFrame({"id": val_df.index}).to_csv(
        REPORT_DATA_DIR / "val_ids.csv",
        index=False,
    )
    pd.DataFrame({"id": test_df.index}).to_csv(
        REPORT_DATA_DIR / "test_ids.csv",
        index=False,
    )

    manifest = {
        "dataset_sha256": calculate_file_sha256(DATA_PATH),
        "random_seed": random_state,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(train_df) + len(val_df) + len(test_df),
    }

    with open(
        REPORT_DATA_DIR / "data_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=4)


def split_data(
    df=None,
    random_state=42,
    use_persisted=True,
    persist_manifest=True,
):
    """Split dataset into train, validation, and test sets.

    If use_persisted is True and persisted ID files exist, loads the persisted splits.
    Otherwise performs stratified splitting and optionally persists the manifest.
    """
    if df is None:
        df = load_data(deduplicate=True)

    train_ids_path = REPORT_DATA_DIR / "train_ids.csv"
    val_ids_path = REPORT_DATA_DIR / "val_ids.csv"
    test_ids_path = REPORT_DATA_DIR / "test_ids.csv"

    if (
        use_persisted
        and train_ids_path.exists()
        and val_ids_path.exists()
        and test_ids_path.exists()
    ):
        train_ids = pd.read_csv(train_ids_path)["id"].values
        val_ids = pd.read_csv(val_ids_path)["id"].values
        test_ids = pd.read_csv(test_ids_path)["id"].values

        # Only use persisted if indices exist in current df
        valid_indices = set(df.index)
        if (
            set(train_ids).issubset(valid_indices)
            and set(val_ids).issubset(valid_indices)
            and set(test_ids).issubset(valid_indices)
        ):
            train_df = df.loc[train_ids].copy()
            val_df = df.loc[val_ids].copy()
            test_df = df.loc[test_ids].copy()
            return train_df, val_df, test_df

    # 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["Topic_group"],
        random_state=random_state,
    )

    # 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Topic_group"],
        random_state=random_state,
    )

    if persist_manifest:
        save_split_manifest(
            train_df,
            val_df,
            test_df,
            random_state=random_state,
        )

    return train_df, val_df, test_df