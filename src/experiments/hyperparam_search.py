"""
experiments/hyperparam_search.py
----------------------------------
FR: Hyperparameter comparison, not just one fixed config.

Trains the LSTM across a small grid of (hidden_size, input_window)
combinations and compares them by pinball loss (the correct metric here,
since we care about the whole predicted distribution, not just the median
-- see FR-5.1 in the SRS).

Kept intentionally small (a 3x2 grid by default) rather than an exhaustive
search -- the point is demonstrating that a search space was considered
and compared fairly, not running an expensive tuning job.

Usage:
    python -m src.experiments.hyperparam_search --dataset rossmann
"""

import argparse
import pandas as pd

from src.pipeline import load_dataset
from src import preprocessing
from src.experiments.common import train_and_evaluate
from src.experiment_log import load_experiment_log
from src.experiments.plots import plot_hyperparam_heatmap

# Small, deliberately limited grid -- see module docstring.
HIDDEN_SIZES = [32, 64, 128]
INPUT_WINDOWS = [14, 28]


def run_hyperparam_search(dataset: str, epochs: int = 6):
    print(f"=== Hyperparameter search on '{dataset}' ===")
    df, feature_cols = load_dataset(dataset)

    results = []
    for hidden_size in HIDDEN_SIZES:
        for input_window in INPUT_WINDOWS:
            tag = f"hs{hidden_size}_iw{input_window}"
            print(f"\n--- Trial: hidden_size={hidden_size}, input_window={input_window} ---")

            data = preprocessing.run_preprocessing_pipeline(
                df=df, feature_cols=feature_cols, input_window=input_window,
            )
            X_train, y_train, _ = data["train"]
            X_val, y_val, _ = data["val"]
            X_test, y_test, _ = data["test"]

            try:
                result = train_and_evaluate(
                    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test,
                    feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
                    epochs=epochs, run_type="hyperparam_search", dataset_name=dataset, tag=tag,
                )
                results.append(result)
            except ValueError as e:
                print(f"Skipped ({e})")

    summary = pd.DataFrame([
        {"hidden_size": r["hidden_size"], "input_window": r["input_window"],
         "pinball_loss": r["pinball_loss"], "mae": r["mae"],
         "empirical_coverage": r["empirical_coverage"], "calibration_error": r["calibration_error"]}
        for r in results
    ]).sort_values("pinball_loss")

    print("\n=== Hyperparameter Search Results (sorted by pinball loss, lower is better) ===")
    print(summary.to_string(index=False))
    print(f"\nBest config: hidden_size={summary.iloc[0]['hidden_size']}, input_window={summary.iloc[0]['input_window']}")
    print("Full results also appended to reports/experiment_log.csv")

    plot_hyperparam_heatmap(summary, metric="pinball_loss")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "m5"], default="rossmann")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    run_hyperparam_search(dataset=args.dataset, epochs=args.epochs)
