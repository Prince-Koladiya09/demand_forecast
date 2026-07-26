"""
experiments/segment_analysis.py
----------------------------------
FR: Per-SKU segmentation of results.

A single aggregate pinball loss across all SKUs can hide very different
behavior for different types of series -- e.g. a model might do well on
high-volume, steady-selling items but poorly on low-volume, intermittent
ones. This script buckets SKUs by historical demand volume (low/medium/
high, via tertiles) and reports metrics PER bucket, not just overall.

This reuses a model trained once via src/pipeline.py's standard single
train/val/test split -- it does not retrain per segment, it only breaks
down evaluation of the SAME trained model's test predictions by SKU group.

Usage:
    python -m src.experiments.segment_analysis --dataset rossmann
"""

import argparse
import numpy as np
import pandas as pd
import torch

from src.pipeline import load_dataset
from src import preprocessing
from src import dl_model
from src import calibration
from src.experiments.common import train_and_evaluate
from src.experiments.plots import plot_segment_results


def assign_volume_buckets(raw_long_df: pd.DataFrame, n_buckets=3):
    """
    Bucket each SKU into volume tertiles based on total historical demand.
    Returns a Series indexed by sku_id -> bucket label ("low", "medium", "high").
    """
    total_volume = raw_long_df.groupby(preprocessing.SKU_COL)[preprocessing.TARGET_COL].sum()
    labels = ["low", "medium", "high"][:n_buckets]
    buckets = pd.qcut(total_volume, q=n_buckets, labels=labels, duplicates="drop")
    return buckets


def evaluate_by_segment(y_test, preds, sku_test, volume_buckets: pd.Series):
    """
    Compute pinball loss, MAE, and empirical coverage separately for each
    volume bucket, using the already-generated test predictions.
    """
    p10, p50, p90 = preds[:, 0, :], preds[:, 1, :], preds[:, 2, :]
    sku_bucket = pd.Series(sku_test).map(volume_buckets)

    rows = []
    for bucket in sorted(volume_buckets.cat.categories, key=lambda b: ["low", "medium", "high"].index(b)):
        mask = (sku_bucket == bucket).values
        if mask.sum() == 0:
            continue

        y_seg, p10_seg, p50_seg, p90_seg = y_test[mask], p10[mask], p50[mask], p90[mask]
        preds_seg = preds[mask]

        pinball = float(dl_model.pinball_loss(
            torch.as_tensor(preds_seg, dtype=torch.float32),
            torch.as_tensor(y_seg, dtype=torch.float32),
        ).item())
        mae_val = float(np.mean(np.abs(y_seg - p50_seg)))
        coverage = calibration.empirical_coverage(y_seg, p10_seg, p90_seg)

        rows.append({
            "volume_bucket": bucket, "n_test_windows": int(mask.sum()),
            "pinball_loss": round(pinball, 4), "mae": round(mae_val, 4),
            "empirical_coverage": round(coverage, 4),
        })

    return pd.DataFrame(rows)


def run_segment_analysis(dataset: str, hidden_size: int = 64, input_window: int = 28, epochs: int = 8):
    print(f"=== Per-SKU Segment Analysis on '{dataset}' ===")
    df, feature_cols = load_dataset(dataset)

    data = preprocessing.run_preprocessing_pipeline(df=df, feature_cols=feature_cols, input_window=input_window)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, sku_test = data["test"]
    train_df, val_df, test_df = data["raw_splits"]

    result = train_and_evaluate(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
        epochs=epochs, run_type="segment_analysis", dataset_name=dataset, tag="overall",
    )

    # Bucket SKUs by volume using the FULL cleaned history (not just test
    # period), since a SKU's overall volume tier is a property of the SKU,
    # not of one time window.
    full_long_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    volume_buckets = assign_volume_buckets(full_long_df)

    segment_results = evaluate_by_segment(y_test, result["preds"], sku_test, volume_buckets)

    print("\n=== Results by Volume Segment ===")
    print(segment_results.to_string(index=False))
    print(f"\nOverall pinball loss (all SKUs combined): {result['pinball_loss']:.4f}")
    print("\nIf pinball loss is much higher for 'low' volume SKUs, that means the model")
    print("struggles more with intermittent/low-volume demand -- a common and expected")
    print("pattern, and worth noting rather than only reporting the blended average.")

    plot_segment_results(segment_results)

    return segment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "m5"], default="rossmann")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run_segment_analysis(dataset=args.dataset, epochs=args.epochs)
