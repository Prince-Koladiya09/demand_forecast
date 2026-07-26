"""
data_loaders/m5_loader.py
--------------------------
Converts the raw M5 Forecasting (Walmart) dataset into the standard long
format expected by src/preprocessing.py.

Download from Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy
You need THREE files in data/raw/:
    - sales_train_validation.csv   (wide format: one column per day, d_1..d_1913)
    - calendar.csv                 (maps d_1, d_2, ... to actual calendar dates + wm_yr_wk)
    - sell_prices.csv              (weekly price per item-store, keyed by wm_yr_wk)

Key adaptation decisions (worth knowing for an interview):
- M5 ships in WIDE format (one column per day). This must be "melted" into
  long format (one row per item-day) before it's usable as a time series
  at all -- this is the main structural difference from Rossmann.
- M5 has ~30,490 item-store series. Training on all of them is unnecessary
  and slow for this project -- we subsample a fixed number of series
  (default 300) to keep training fast WITHOUT reducing history length
  per series (each kept series still has its full ~1,913-day history).
- sell_prices.csv is keyed by (store_id, item_id, wm_yr_wk) -- a WEEKLY
  price, not daily. It's merged onto each day via that day's wm_yr_wk
  (from calendar.csv), so the same weekly price is repeated across each
  day in that week. Price is a genuine demand driver (promotions/price
  changes shift sales) and is carried through as an extra input channel,
  not just an unused column.
- Some item-weeks have no recorded price (item not sold/stocked yet).
  Those are forward/backward filled per series in
  src/preprocessing.py's fill_missing(), since a missing price almost
  always means "same as the nearest known price", not "price of zero".
"""

import pandas as pd
from pathlib import Path

RAW_SALES_PATH = Path("data/raw/sales_train_validation.csv")
RAW_CALENDAR_PATH = Path("data/raw/calendar.csv")
RAW_PRICES_PATH = Path("data/raw/sell_prices.csv")

# Extra numeric feature columns this loader produces, for use as additional
# input channels in src/preprocessing.py's make_windows(feature_cols=...).
FEATURE_COLS = ["sell_price"]


def load_m5(
    sales_path: Path = RAW_SALES_PATH,
    calendar_path: Path = RAW_CALENDAR_PATH,
    prices_path: Path = RAW_PRICES_PATH,
    n_series: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: date, sku_id, units_sold, sell_price
    matching the schema expected by src/preprocessing.py.
    """
    sales_wide = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    # Subsample series to keep training fast -- full history length per
    # series is preserved, only the NUMBER of series is reduced.
    if n_series is not None and n_series < len(sales_wide):
        sales_wide = sales_wide.sample(n=n_series, random_state=random_state)

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    # Melt from wide (one column per day) to long (one row per item-day).
    # Keep item_id/store_id (already columns in sales_train_validation.csv)
    # so we can later join sell_prices.csv, which is keyed on those.
    id_vars = ["id", "item_id", "store_id"]
    long_df = sales_wide.melt(
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )

    # Map "d_1", "d_2", ... to actual calendar dates AND to wm_yr_wk
    # (the weekly id that sell_prices.csv is keyed on).
    calendar_lookup = calendar.set_index("d")[["date", "wm_yr_wk"]]
    long_df = long_df.merge(calendar_lookup, left_on="d", right_index=True, how="left")

    # Merge weekly price onto each day via (store_id, item_id, wm_yr_wk).
    # The same weekly price will repeat across every day in that week --
    # this is expected, since M5 prices are only recorded at weekly granularity.
    long_df = long_df.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )

    long_df = long_df.rename(columns={"id": "sku_id"})
    long_df = long_df[["date", "sku_id", "units_sold", "sell_price"]]
    long_df = long_df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    return long_df


if __name__ == "__main__":
    df = load_m5(n_series=50)  # small subsample for a quick smoke test
    print(df.head())
    print("\nSeries kept:", df["sku_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Missing sell_price rows:", df["sell_price"].isna().sum(), "/", len(df))
    print("Feature columns available:", FEATURE_COLS)
