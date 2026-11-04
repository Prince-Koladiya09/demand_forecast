
"""
data_loaders/m5_loader.py

Converts the raw M5 Forecasting dataset into long format.

Required files in data/raw/:
    - sales_train_validation.csv
    - calendar.csv
    - sell_prices.csv
"""

from pathlib import Path

import pandas as pd


RAW_SALES_PATH = Path("data/raw/sales_train_validation.csv")
RAW_CALENDAR_PATH = Path("data/raw/calendar.csv")
RAW_PRICES_PATH = Path("data/raw/sell_prices.csv")

FEATURE_COLS = ["sell_price"]


def load_m5(
    sales_path: Path = RAW_SALES_PATH,
    calendar_path: Path = RAW_CALENDAR_PATH,
    prices_path: Path = RAW_PRICES_PATH,
    n_series: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame with:
        date, sku_id, units_sold, sell_price

    M5's wide daily sales data is converted to long format.
    Weekly prices are merged using store_id, item_id and wm_yr_wk.
    """

    sales_wide = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    # Validate required columns.
    required_sales = {"id", "item_id", "store_id"}
    required_calendar = {"d", "date", "wm_yr_wk"}
    required_prices = {"store_id", "item_id", "wm_yr_wk", "sell_price"}

    if not required_sales.issubset(sales_wide.columns):
        raise ValueError(
            f"Sales file is missing columns: "
            f"{required_sales - set(sales_wide.columns)}"
        )

    if not required_calendar.issubset(calendar.columns):
        raise ValueError(
            f"Calendar file is missing columns: "
            f"{required_calendar - set(calendar.columns)}"
        )

    if not required_prices.issubset(prices.columns):
        raise ValueError(
            f"Price file is missing columns: "
            f"{required_prices - set(prices.columns)}"
        )

    # Subsample series while preserving their complete history.
    if n_series is not None and n_series < len(sales_wide):
        sales_wide = sales_wide.sample(
            n=n_series,
            random_state=random_state,
        )

    day_cols = [
        column
        for column in sales_wide.columns
        if column.startswith("d_")
    ]

    if not day_cols:
        raise ValueError("No daily sales columns (d_1, d_2, ...) found.")

    # Convert wide daily sales data into long format.
    long_df = sales_wide.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )

    # Map day identifiers to calendar dates and week identifiers.
    calendar_lookup = calendar[["d", "date", "wm_yr_wk"]]

    long_df = long_df.merge(
        calendar_lookup,
        on="d",
        how="left",
        validate="many_to_one",
    )

    # Merge weekly prices onto daily observations.
    long_df = long_df.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    long_df = long_df.rename(columns={"id": "sku_id"})

    # Keep only observations from the first active date for each series.
    active = (
        long_df["sell_price"].notna()
        | (long_df["units_sold"].fillna(0) > 0)
    )

    first_active = (
        long_df.loc[active]
        .groupby("sku_id")["date"]
        .min()
    )

    long_df["_first_active"] = long_df["sku_id"].map(first_active)

    long_df = long_df[
        long_df["_first_active"].isna()
        | (long_df["date"] >= long_df["_first_active"])
    ].copy()

    # Final output.
    long_df = long_df[
        ["date", "sku_id", "units_sold", "sell_price"]
    ]

    long_df["units_sold"] = pd.to_numeric(
        long_df["units_sold"],
        errors="coerce",
    )

    long_df["sell_price"] = pd.to_numeric(
        long_df["sell_price"],
        errors="coerce",
    )

    long_df = long_df.sort_values(
        ["sku_id", "date"]
    ).reset_index(drop=True)

    return long_df


if __name__ == "__main__":
    df = load_m5(n_series=50)

    print(df.head())
    print("\nSeries kept:", df["sku_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print(
        "Missing sell_price rows:",
        df["sell_price"].isna().sum(),
        "/",
        len(df),
    )
    print("Feature columns available:", FEATURE_COLS)
