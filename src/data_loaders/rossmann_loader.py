"""
data_loaders/rossmann_loader.py
--------------------------------
Converts the raw Rossmann Store Sales files into the standard long format
expected by src/preprocessing.py:

    date, sku_id, units_sold, day_of_week, promo, competition_distance,
    promo2, store_type_code, assortment_code

Download from Kaggle: https://www.kaggle.com/c/rossmann-store-sales
Place `train.csv` AND `store.csv` in data/raw/.

Key adaptation decisions (worth knowing for an interview):
- "Store" is treated as the SKU/series id, since Rossmann is store-level,
  not item-level, demand.
- Rows where Open == 0 (store closed) are DROPPED rather than treated as
  zero-demand days. A closed store isn't "zero demand" in the same sense
  as an open store selling nothing -- keeping them in would bias the
  model into thinking closures are a demand pattern to predict.
- store.csv adds store-level context (StoreType, Assortment,
  CompetitionDistance, Promo2) that plain historical sales can't capture
  on their own -- e.g. two stores with similar past sales can diverge
  once a competitor opens nearby. These are merged in as extra numeric
  input channels for the DL model (see FEATURE_COLS below), not just
  carried along unused.
- StoreType and Assortment are categorical (a/b/c/d letters in the raw
  data) and are label-encoded here into integers so they can be used as
  numeric model inputs. This is a deliberately simple encoding (not
  one-hot) to keep the model input size small -- reasonable given there
  are only a handful of categories.
"""

import pandas as pd
from pathlib import Path

RAW_TRAIN_PATH = Path("data/raw/train.csv")
RAW_STORE_PATH = Path("data/raw/store.csv")

# Extra numeric feature columns this loader produces, for use as additional
# input channels in src/preprocessing.py's make_windows(feature_cols=...).
FEATURE_COLS = ["promo", "competition_distance", "promo2", "store_type_code", "assortment_code"]


def load_rossmann(
    train_path: Path = RAW_TRAIN_PATH,
    store_path: Path = RAW_STORE_PATH,
    min_history_days: int = 100,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        date, sku_id, units_sold, day_of_week, promo,
        competition_distance, promo2, store_type_code, assortment_code
    matching the schema expected by src/preprocessing.py
    (DATE_COL="date", SKU_COL="sku_id", TARGET_COL="units_sold").
    """
    df = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)

    # Drop days the store was closed -- these are not informative zero-demand days
    df = df[df["Open"] == 1].copy()

    df = df.rename(columns={
        "Date": "date",
        "Store": "sku_id",
        "Sales": "units_sold",
        "DayOfWeek": "day_of_week",
        "Promo": "promo",
    })

    # --- Merge store-level metadata ---
    store_df = pd.read_csv(store_path)
    store_df = store_df.rename(columns={"Store": "sku_id"})

    # Label-encode categorical columns into small integer codes
    store_df["store_type_code"] = store_df["StoreType"].astype("category").cat.codes
    store_df["assortment_code"] = store_df["Assortment"].astype("category").cat.codes

    # CompetitionDistance can be missing (no competitor reported) -- fill
    # with the max observed distance, treating "unknown" as "far away"
    # rather than 0 (which would wrongly mean "competitor next door").
    max_dist = store_df["CompetitionDistance"].max()
    store_df["competition_distance"] = store_df["CompetitionDistance"].fillna(max_dist)

    # Promo2 (0/1): whether the store runs recurring promotions
    store_df["promo2"] = store_df["Promo2"].fillna(0)

    store_features = store_df[["sku_id", "competition_distance", "promo2", "store_type_code", "assortment_code"]]

    df = df.merge(store_features, on="sku_id", how="left")

    df = df[["date", "sku_id", "units_sold", "day_of_week", "promo"] + FEATURE_COLS[1:]]
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # Drop stores with too little history to build meaningful windows
    counts = df.groupby("sku_id")["date"].count()
    valid_stores = counts[counts >= min_history_days].index
    df = df[df["sku_id"].isin(valid_stores)].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_rossmann()
    print(df.head())
    print("\nStores kept:", df["sku_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Feature columns available:", FEATURE_COLS)
