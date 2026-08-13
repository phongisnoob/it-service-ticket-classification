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


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[["Document", "Topic_group"]].copy()

    df = df.dropna()

    df["Document"] = df["Document"].astype(str)
    df["Topic_group"] = df["Topic_group"].astype(str)

    return df


def split_data(df, random_state=42):

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

    return train_df, val_df, test_df