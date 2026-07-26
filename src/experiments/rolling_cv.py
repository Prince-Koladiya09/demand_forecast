"""
experiments/rolling_cv.py
---------------------------
FR: Cross-validation across time windows, not just one train/val/test split.

A single train/val/test split (what src/pipeline.py does) only tells you
how the model did on ONE particular test period. That period could have
been unusually easy or hard by chance. Rolling-origin cross-validation
(a.k.a. walk-forward validation) fixes this by evaluating across several
different time periods and looking at the SPREAD of results, not just one
number.

Standard random k-fold CV is NOT valid for time series -- it would let the
model train on future data and test on the past, which is leakage. Rolling-
origin CV instead expands the training window forward in time across folds,
always testing on the period immediately after training ends:

    Fold 1: train [0.................|60%]  test [60%-70%]
    Fold 2: train [0.....................|70%]  test [70%-80%]
    Fold 3: train [0.........................|80%]  test [80%-90%]

Usage:
    python -m src.experiments.rolling_cv --dataset rossmann --n_folds 3
"""

import argparse
import numpy as np
import pandas as pd

from src.pipeline import load_dataset
from src import preprocessing
from src.experiments.common import train_and_evaluate
from src.experiments.plots import plot_rolling_cv_folds


def rolling_origin_split(df: pd.DataFrame, n_folds=3, test_frac_per_fold=0.10, min_train_frac=0.5):
    """
    Build n_folds (train_df, test_df) pairs per SKU, expanding the training
    window forward in time across folds. Returns a list of (train_df, test_df).
    """
    folds = []
    for fold_idx in range(n_folds):
        train_end_frac = min_train_frac + fold_idx * test_frac_per_fold
        test_end_frac = train_end_frac + test_frac_per_fold
        if test_end_frac > 1.0:
            break

        train_frames, test_frames = [], []
        for sku, group in df.groupby(preprocessing.SKU_COL):
            n = len(group)
            train_end = int(n * train_end_frac)
            test_end = int(n * test_end_frac)
            train_frames.append(group.iloc[:train_end])
            test_frames.append(group.iloc[train_end:test_end])

        train_df = pd.concat(train_frames, ignore_index=True)
        test_df = pd.concat(test_frames, ignore_index=True)
        folds.append((train_df, test_df))

    return folds


def run_rolling_cv(dataset: str, n_folds: int = 3, hidden_size: int = 64, input_window: int = 28, epochs: int = 6):
    print(f"=== Rolling-Origin Cross-Validation on '{dataset}' ({n_folds} folds) ===")
    df, feature_cols = load_dataset(dataset)

    loaded_df = df.sort_values([preprocessing.SKU_COL, preprocessing.DATE_COL]).reset_index(drop=True)
    loaded_df = preprocessing.fill_missing(loaded_df, feature_cols=feature_cols)
    loaded_df = preprocessing.add_calendar_features(loaded_df)

    folds = rolling_origin_split(loaded_df, n_folds=n_folds)

    results = []
    for i, (train_df, test_df) in enumerate(folds):
        print(f"\n--- Fold {i + 1}/{len(folds)} ---")
        # Use the fold's own training data as both train+val (small internal
        # split) since rolling CV already limits how much history is available
        # per fold -- carving out yet another val split would leave too little
        # data. We split the fold's train_df 85/15 for train/val internally.
        val_frames, train_frames = [], []
        for sku, group in train_df.groupby(preprocessing.SKU_COL):
            n_val = max(1, int(len(group) * 0.15))
            train_frames.append(group.iloc[:-n_val] if n_val < len(group) else group)
            val_frames.append(group.iloc[-n_val:])
        inner_train_df = pd.concat(train_frames, ignore_index=True)
        inner_val_df = pd.concat(val_frames, ignore_index=True)

        try:
            result = train_and_evaluate(
                train_df=inner_train_df, val_df=inner_val_df, test_df=test_df,
                feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
                epochs=epochs, run_type="rolling_cv", dataset_name=dataset, tag=f"fold_{i+1}", fold=i + 1,
            )
            results.append(result)
        except ValueError as e:
            print(f"Skipped fold {i + 1} ({e})")

    pinball_scores = [r["pinball_loss"] for r in results]
    coverage_scores = [r["empirical_coverage"] for r in results]

    print("\n=== Rolling CV Summary ===")
    print(f"Folds completed: {len(results)}")
    print(f"Pinball loss per fold: {pinball_scores}")
    print(f"Mean pinball loss: {np.mean(pinball_scores):.4f}  (std: {np.std(pinball_scores):.4f})")
    print(f"Mean empirical coverage: {np.mean(coverage_scores):.3f}  (std: {np.std(coverage_scores):.3f})")
    print("\nA high std across folds means performance is unstable across time periods --")
    print("worth knowing before trusting a single train/val/test split's result.")

    if len(results) >= 2:
        plot_rolling_cv_folds(pinball_scores, coverage_scores)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "m5"], default="rossmann")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    run_rolling_cv(dataset=args.dataset, n_folds=args.n_folds, epochs=args.epochs)
