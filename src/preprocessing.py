"""
preprocessing.py
----------------
Implements FR-1: Data Ingestion & Preprocessing.

Responsibilities:
- Load raw sales data
- Handle missing values / zero-demand days WITHOUT dropping them
  (zeros are informative in retail demand, not noise)
- Build sliding-window sequences for supervised learning
- Perform a CHRONOLOGICAL train/val/test split (never random shuffle
  for time series -- that would leak future information into training)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---- Update this to point at your dataset ----
RAW_DATA_PATH = Path("data/raw/sales.csv")

# Columns expected in the raw CSV. Adjust to match your dataset
# (e.g., M5 requires melting from wide to long format first).
DATE_COL = "date"
SKU_COL = "sku_id"
TARGET_COL = "units_sold"

INPUT_WINDOW = 28   # days of history used as input
OUTPUT_WINDOW = 7    # days of future demand to forecast


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw sales data. Expects long format: one row per (date, sku)."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values([SKU_COL, DATE_COL]).reset_index(drop=True)
    return df


def fill_missing(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """
    Fill missing calendar days per SKU with 0 demand (a missing row usually
    means no sale happened, not that data is unavailable). Explicit zeros
    are kept -- they are informative for intermittent-demand items.

    Any extra feature columns (e.g. sell_price, competition_distance) are
    forward-filled then back-filled instead of zeroed out -- a missing
    price on a reindexed date almost certainly means "same price as the
    nearest known week", not "price of zero".
    """
    feature_cols = feature_cols or []
    filled_frames = []
    for sku, group in df.groupby(SKU_COL):
        full_range = pd.date_range(group[DATE_COL].min(), group[DATE_COL].max(), freq="D")
        group = group.set_index(DATE_COL).reindex(full_range)
        group[SKU_COL] = sku
        group[TARGET_COL] = group[TARGET_COL].fillna(0)
        for col in feature_cols:
            if col in group.columns:
                group[col] = group[col].ffill().bfill().fillna(0)
        group = group.rename_axis(DATE_COL).reset_index()
        filled_frames.append(group)
    return pd.concat(filled_frames, ignore_index=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar features (day-of-week, week-of-year)."""
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    return df


def chronological_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    """
    Split EACH sku's series chronologically. Never shuffle randomly --
    that would let the model see the "future" during training, which is a
    common and serious leakage bug in time-series projects.
    """
    train_frames, val_frames, test_frames = [], [], []
    for sku, group in df.groupby(SKU_COL):
        n = len(group)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test

        train_frames.append(group.iloc[:n_train])
        val_frames.append(group.iloc[n_train:n_train + n_val])
        test_frames.append(group.iloc[n_train + n_val:])

    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(val_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True),
    )


def make_windows(df: pd.DataFrame, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, feature_cols=None):
    """
    Convert a per-SKU time series into (X, y) sliding window samples.

    X: shape (n_samples, input_window, n_channels) where channel 0 is always
       the target (units_sold) and any subsequent channels are extra
       numeric features (e.g. sell_price, promo, competition_distance)
       passed in via `feature_cols`. Keeping the target as channel 0 means
       downstream code (e.g. the SARIMA baseline) can always recover the
       plain demand series with X[..., 0].
    y: shape (n_samples, output_window) -- always just the target, since
       we are forecasting demand, not the extra features themselves.

    feature_cols: list of extra numeric column names to include as input
       channels (e.g. ["sell_price"] for M5, ["promo", "competition_distance",
       "promo2", "store_type_code", "assortment_code"] for Rossmann).
       Leave as None/[] to reproduce the original single-channel behavior.
    """
    feature_cols = feature_cols or []
    channel_cols = [TARGET_COL] + feature_cols

    X_list, y_list, sku_list = [], [], []

    for sku, group in df.groupby(SKU_COL):
        values = group[channel_cols].values  # shape (n_days, n_channels)
        target = group[TARGET_COL].values
        n = len(values)
        for start in range(0, n - input_window - output_window + 1):
            x_window = values[start: start + input_window, :]                       # (input_window, n_channels)
            y_window = target[start + input_window: start + input_window + output_window]  # (output_window,)
            X_list.append(x_window)
            y_list.append(y_window)
            sku_list.append(sku)

    X = np.array(X_list)  # (n_samples, input_window, n_channels)
    y = np.array(y_list)  # (n_samples, output_window)
    return X, y, np.array(sku_list)


PROCESSED_DIR = Path("data/processed")


def save_processed_data(data: dict, dataset_name: str, processed_dir: Path = PROCESSED_DIR):
    """
    Cache the output of run_preprocessing_pipeline() to data/processed/ so
    that later runs (training experiments, the EDA notebook, etc.) don't
    need to re-run preprocessing from raw CSVs every time -- preprocessing
    (especially M5's melt/merge) is the slow part; training/experimenting
    is what you actually do repeatedly.

    Saves:
    - {dataset_name}_train.npz / _val.npz / _test.npz  -> X, y, sku arrays
    - {dataset_name}_raw_long.parquet                   -> cleaned long-format
      DataFrame (post fill_missing/add_calendar_features, pre-windowing) --
      this is what the EDA notebook loads for plotting, since it's much
      easier to plot a normal DataFrame than windowed arrays.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X, y, sku = data[split_name]
        np.savez(
            processed_dir / f"{dataset_name}_{split_name}.npz",
            X=X, y=y, sku=sku,
        )

    train_df, val_df, test_df = data["raw_splits"]
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_parquet(processed_dir / f"{dataset_name}_cleaned_long.parquet", index=False)

    print(f"Saved processed data to {processed_dir}/ (prefix: '{dataset_name}_')")


def load_processed_windows(dataset_name: str, split: str, processed_dir: Path = PROCESSED_DIR):
    """Load a previously cached (X, y, sku) split without re-running preprocessing."""
    npz = np.load(processed_dir / f"{dataset_name}_{split}.npz", allow_pickle=True)
    return npz["X"], npz["y"], npz["sku"]


def load_processed_long_df(dataset_name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned long-format DataFrame (used by the EDA notebook)."""
    return pd.read_parquet(processed_dir / f"{dataset_name}_cleaned_long.parquet")


def run_preprocessing_pipeline(path: Path = RAW_DATA_PATH, df: pd.DataFrame = None, feature_cols=None,
                                save_as: str = None, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW):
    """
    Convenience entry point used by src/pipeline.py

    Two ways to call this:
    1. run_preprocessing_pipeline(path=...) -- reads a CSV already in the
       standard long format (date, sku_id, units_sold) directly.
    2. run_preprocessing_pipeline(df=already_loaded_df) -- pass in a
       DataFrame produced by a dataset-specific loader
       (src/data_loaders/rossmann_loader.py or m5_loader.py), which have
       already converted a raw dataset into the standard schema.
       This is the recommended path for Rossmann/M5.

    feature_cols: extra numeric columns present in `df` to include as
       additional input channels (e.g. ["sell_price"] for M5). These must
       already exist as columns in `df` -- this function does not create
       them, it only carries them through filling/windowing.

    input_window / output_window: override the default window sizes --
       used by src/experiments/hyperparam_search.py to compare different
       history lengths without editing this file.

    save_as: if given (e.g. "rossmann" or "m5"), caches the result to
       data/processed/ via save_processed_data() so it can be reloaded
       later without re-running preprocessing.
    """
    feature_cols = feature_cols or []

    if df is None:
        loaded_df = load_raw_data(path)
    else:
        loaded_df = df.sort_values([SKU_COL, DATE_COL]).reset_index(drop=True)

    loaded_df = fill_missing(loaded_df, feature_cols=feature_cols)
    loaded_df = add_calendar_features(loaded_df)

    train_df, val_df, test_df = chronological_split(loaded_df)

    X_train, y_train, sku_train = make_windows(train_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)
    X_val, y_val, sku_val = make_windows(val_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)
    X_test, y_test, sku_test = make_windows(test_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)

    result = {
        "train": (X_train, y_train, sku_train),
        "val": (X_val, y_val, sku_val),
        "test": (X_test, y_test, sku_test),
        "raw_splits": (train_df, val_df, test_df),
        "feature_cols": feature_cols,
    }

    if save_as:
        save_processed_data(result, dataset_name=save_as)

    return result


if __name__ == "__main__":
    data = run_preprocessing_pipeline()
    print("Train windows:", data["train"][0].shape)
    print("Val windows:  ", data["val"][0].shape)
    print("Test windows: ", data["test"][0].shape)
