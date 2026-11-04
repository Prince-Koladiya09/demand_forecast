
"""
data_loaders/rossmann_loader.py

Converts raw Rossmann Store Sales files into long format.

Required files in data/raw/:
    - train.csv
    - store.csv

Output columns:
    date, sku_id, units_sold, day_of_week, promo,
    competition_distance, promo2, store_type_code,
    assortment_code, is_open
"""

from pathlib import Path

import pandas as pd


RAW_TRAIN_PATH = Path("data/raw/train.csv")
RAW_STORE_PATH = Path("data/raw/store.csv")

FEATURE_COLS = [
    "promo",
    "competition_distance",
    "promo2",
    "store_type_code",
    "assortment_code",
    "is_open",
]


def load_rossmann(
    train_path: Path = RAW_TRAIN_PATH,
    store_path: Path = RAW_STORE_PATH,
    min_history_days: int = 100,
) -> pd.DataFrame:
    """
    Return a DataFrame with:
        date, sku_id, units_sold, day_of_week, promo,
        competition_distance, promo2, store_type_code,
        assortment_code, is_open

    Store is treated as the time-series identifier.
    """

    # Load training data.
    df = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        low_memory=False,
    )

    required_train = {
        "Date",
        "Store",
        "Sales",
        "DayOfWeek",
        "Promo",
        "Open",
    }

    if not required_train.issubset(df.columns):
        raise ValueError(
            f"Rossmann train file is missing columns: "
            f"{required_train - set(df.columns)}"
        )

    # Preserve the original open/closed indicator.
    df["is_open"] = pd.to_numeric(
        df["Open"],
        errors="coerce",
    ).fillna(1).astype(int)

    # Rename columns to the common project schema.
    df = df.rename(
        columns={
            "Date": "date",
            "Store": "sku_id",
            "Sales": "units_sold",
            "DayOfWeek": "day_of_week",
            "Promo": "promo",
        }
    )

    # Convert numeric fields.
    df["units_sold"] = pd.to_numeric(
        df["units_sold"],
        errors="coerce",
    )

    df["promo"] = pd.to_numeric(
        df["promo"],
        errors="coerce",
    ).fillna(0)

    df["day_of_week"] = pd.to_numeric(
        df["day_of_week"],
        errors="coerce",
    )

    # Closed days are excluded from the demand training data.
    df = df[df["is_open"] == 1].copy()

    # Remove rows with missing sales or dates.
    df = df.dropna(
        subset=["date", "sku_id", "units_sold"]
    ).copy()

    # Load store metadata.
    store_df = pd.read_csv(store_path)

    required_store = {
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Promo2",
    }

    if not required_store.issubset(store_df.columns):
        raise ValueError(
            f"Rossmann store file is missing columns: "
            f"{required_store - set(store_df.columns)}"
        )

    store_df = store_df.rename(
        columns={"Store": "sku_id"}
    )

    # Encode categorical store features.
    store_df["store_type_code"] = (
        store_df["StoreType"].astype("category").cat.codes
    )

    store_df["assortment_code"] = (
        store_df["Assortment"].astype("category").cat.codes
    )

    # Fill missing competition distance with maximum observed distance.
    store_df["CompetitionDistance"] = pd.to_numeric(
        store_df["CompetitionDistance"],
        errors="coerce",
    )

    max_dist = store_df["CompetitionDistance"].max()

    if pd.isna(max_dist):
        max_dist = 0

    store_df["competition_distance"] = (
        store_df["CompetitionDistance"].fillna(max_dist)
    )

    # Promo2 is a store-level indicator.
    store_df["promo2"] = pd.to_numeric(
        store_df["Promo2"],
        errors="coerce",
    ).fillna(0)

    store_features = store_df[
        [
            "sku_id",
            "competition_distance",
            "promo2",
            "store_type_code",
            "assortment_code",
        ]
    ]

    # Merge store-level features.
    df = df.merge(
        store_features,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    # Drop stores with insufficient history.
    counts = df.groupby("sku_id")["date"].nunique()

    valid_stores = counts[
        counts >= min_history_days
    ].index

    df = df[
        df["sku_id"].isin(valid_stores)
    ].copy()

    # Final output columns.
    output_cols = [
        "date",
        "sku_id",
        "units_sold",
        "day_of_week",
        *FEATURE_COLS,
    ]

    df = df[output_cols]

    df = df.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_rossmann()

    print(df.head())
    print("\nStores kept:", df["sku_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Feature columns available:", FEATURE_COLS)
