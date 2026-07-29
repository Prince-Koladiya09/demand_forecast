"""
experiments/advanced_analysis.py
------------------------------------
Ties together three advanced analyses on top of the main pipeline's trained
model and data:

1. Uncertainty Decomposition (src/uncertainty_decomposition.py) -- splits
   predictive uncertainty into epistemic (model) vs. aleatoric (data) via
   MC Dropout, and reports how each changes across the forecast horizon.

2. Structural Break Detection (src/structural_break_detection.py) -- runs
   CUSUM break detection per SKU, then compares model performance WITH vs.
   WITHOUT pruning pre-break training data.

3. Hierarchical Reconciliation (src/hierarchical_reconciliation.py) -- takes
   each SKU's most recent forecast plus an INDEPENDENTLY fit SARIMA model
   on the aggregated total demand, and reconciles the two so they agree.

Each section logs to reports/experiment_log.csv and saves a chart to
reports/figures/, consistent with the other experiment scripts.

Usage:
    python -m src.experiments.advanced_analysis --dataset rossmann
"""

import argparse
import numpy as np
import pandas as pd

from src.pipeline import load_dataset
from src import preprocessing
from src import baseline_sarima
from src.experiments.common import train_and_evaluate
from src.experiment_log import log_experiment
from src.experiments.plots import (
    plot_uncertainty_decomposition,
    plot_regime_pruning_comparison,
    plot_reconciliation_comparison,
)
from src.uncertainty_decomposition import mc_dropout_predict, summarize_by_horizon
from src.structural_break_detection import detect_breaks_per_sku, apply_regime_pruning
from src.hierarchical_reconciliation import reconcile_forecast_matrix


def run_uncertainty_decomposition(model, X_test, dataset_name, n_passes=20):
    print("\n--- Uncertainty Decomposition (MC Dropout) ---")
    mean_preds, epistemic_std, aleatoric_spread = mc_dropout_predict(model, X_test, n_passes=n_passes)
    summary = summarize_by_horizon(epistemic_std, aleatoric_spread)

    print("Epistemic uncertainty by horizon (t+1..t+N):", np.round(summary["epistemic_by_horizon"], 3))
    print("Aleatoric uncertainty by horizon (t+1..t+N):", np.round(summary["aleatoric_by_horizon"], 3))

    mean_epistemic = float(summary["epistemic_by_horizon"].mean())
    mean_aleatoric = float(summary["aleatoric_by_horizon"].mean())
    dominant = "epistemic (model uncertainty)" if mean_epistemic > mean_aleatoric else "aleatoric (data noise)"
    print(f"On average, {dominant} dominates total predictive uncertainty for this dataset/model.")

    log_experiment({
        "run_type": "uncertainty_decomposition", "dataset": dataset_name, "tag": "mc_dropout",
        "notes": f"mean_epistemic={mean_epistemic:.4f}, mean_aleatoric={mean_aleatoric:.4f}, dominant={dominant}",
    })

    plot_uncertainty_decomposition(summary)
    return summary


def run_structural_break_analysis(dataset_name, train_df, val_df, test_df, feature_cols,
                                   hidden_size, input_window, epochs, baseline_result):
    print("\n--- Structural Break Detection (CUSUM) + Regime Pruning ---")
    full_long_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    breaks = detect_breaks_per_sku(
        full_long_df, sku_col=preprocessing.SKU_COL, date_col=preprocessing.DATE_COL,
        target_col=preprocessing.TARGET_COL,
    )
    n_breaks = sum(1 for v in breaks.values() if v is not None)
    print(f"Detected structural breaks in {n_breaks}/{len(breaks)} SKUs.")

    if n_breaks == 0:
        print("No breaks detected -- skipping regime-pruning comparison (nothing to prune).")
        return None

    pruned_df = apply_regime_pruning(full_long_df, preprocessing.SKU_COL, preprocessing.DATE_COL, breaks)
    print(f"Rows before pruning: {len(full_long_df)}, after pruning: {len(pruned_df)}")

    pruned_train_df, pruned_val_df, pruned_test_df = preprocessing.chronological_split(pruned_df)

    try:
        pruned_result = train_and_evaluate(
            train_df=pruned_train_df, val_df=pruned_val_df, test_df=pruned_test_df,
            feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
            epochs=epochs, run_type="structural_break_analysis", dataset_name=dataset_name,
            tag="with_regime_pruning",
        )
    except ValueError as e:
        print(f"Skipped regime-pruning comparison ({e})")
        return None

    summary_df = pd.DataFrame([
        {"tag": "no_pruning", "pinball_loss": baseline_result["pinball_loss"]},
        {"tag": "with_regime_pruning", "pinball_loss": pruned_result["pinball_loss"]},
    ])
    print("\n=== Regime Pruning Comparison ===")
    print(summary_df.to_string(index=False))

    if pruned_result["pinball_loss"] < baseline_result["pinball_loss"]:
        print("Regime pruning IMPROVED pinball loss -- pre-break data was adding noise/bias to training.")
    else:
        print("Regime pruning did NOT improve pinball loss on this data/config -- "
              "the detected break(s) may not be large enough to outweigh the lost training history.")

    plot_regime_pruning_comparison(summary_df)
    return summary_df


def run_hierarchical_reconciliation(dataset_name, train_df, sku_test, p50, output_window):
    print("\n--- Hierarchical Reconciliation ---")

    # Take each SKU's MOST RECENT test-window forecast as its "current" forecast.
    # NOTE: this assumes forecast windows are reasonably aligned in time across
    # SKUs (true when all series share the same date range, as in this project's
    # chronological_split) -- a simplification stated explicitly here, not a
    # hidden assumption. A production system would instead generate all
    # per-SKU forecasts from the SAME origin date explicitly.
    if sku_test is None:
        print("No per-window SKU labels available (train_df/val_df/test_df path was used) -- skipping reconciliation.")
        return None

    last_idx_per_sku = {}
    for i, sku in enumerate(sku_test):
        last_idx_per_sku[sku] = i  # later occurrences overwrite -> ends up as the LAST window per SKU

    skus_sorted = sorted(last_idx_per_sku.keys())
    if len(skus_sorted) < 2:
        print("Fewer than 2 SKUs available -- reconciliation needs at least 2 bottom-level series. Skipping.")
        return None

    bottom_forecast_matrix = np.stack([p50[last_idx_per_sku[sku]] for sku in skus_sorted])  # (n_series, output_window)

    # Bottom-level variance proxy: variance of each SKU's own training-period demand
    bottom_variances = np.array([
        train_df[train_df[preprocessing.SKU_COL] == sku][preprocessing.TARGET_COL].var() + 1e-6
        for sku in skus_sorted
    ])

    # Independent top-level forecast: fit SARIMA directly on the AGGREGATED
    # total demand (summed across all SKUs per day) -- a genuinely separate
    # model from the per-SKU LSTM, which is what makes reconciliation
    # meaningful rather than a no-op.
    total_series = (
        train_df.groupby(preprocessing.DATE_COL)[preprocessing.TARGET_COL]
        .sum()
        .sort_index()
        .values
    )
    try:
        sarima_fitted = baseline_sarima.fit_sarima(total_series)
        top_forecast_series = baseline_sarima.forecast_sarima(sarima_fitted, steps=output_window)["mean"]
    except Exception as e:
        print(f"SARIMA fit on aggregated total demand failed ({e}) -- skipping reconciliation.")
        return None

    matrix_result = reconcile_forecast_matrix(
        bottom_forecast_matrix, bottom_variances, top_forecast_series=top_forecast_series,
    )

    print(f"Reconciling {len(skus_sorted)} SKUs against an independent total-demand SARIMA forecast.")
    print("Raw bottom-up sum per horizon step:      ", np.round(matrix_result["top_raw"], 2))
    print("Independent top-level forecast per step: ", np.round(np.asarray(top_forecast_series), 2))
    print("Reconciled total per horizon step:       ", np.round(matrix_result["top_reconciled"], 2))

    log_experiment({
        "run_type": "hierarchical_reconciliation", "dataset": dataset_name, "tag": "mint_structural_scaling",
        "n_features": len(skus_sorted),
        "notes": f"mean_raw_sum={matrix_result['top_raw'].mean():.2f}, "
                 f"mean_independent_top={np.mean(top_forecast_series):.2f}, "
                 f"mean_reconciled={matrix_result['top_reconciled'].mean():.2f}",
    })

    plot_reconciliation_comparison(matrix_result["top_raw"], top_forecast_series, matrix_result["top_reconciled"])
    return matrix_result


def run_advanced_analysis(dataset: str, hidden_size: int = 64, input_window: int = 28, epochs: int = 8):
    print(f"=== Advanced Analysis on '{dataset}' ===")
    df, feature_cols = load_dataset(dataset)

    data = preprocessing.run_preprocessing_pipeline(df=df, feature_cols=feature_cols, input_window=input_window)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, sku_test = data["test"]
    train_df, val_df, test_df = data["raw_splits"]

    baseline_result = train_and_evaluate(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
        epochs=epochs, run_type="advanced_analysis", dataset_name=dataset, tag="base_model",
    )
    model = baseline_result["model"]

    uncertainty_summary = run_uncertainty_decomposition(model, X_test, dataset)

    regime_summary = run_structural_break_analysis(
        dataset, train_df, val_df, test_df, feature_cols, hidden_size, input_window, epochs, baseline_result,
    )

    reconciliation_result = run_hierarchical_reconciliation(
        dataset, train_df, sku_test, baseline_result["p50"], y_test.shape[1],
    )

    print("\n=== Advanced Analysis Complete ===")
    print("Results logged to reports/experiment_log.csv")
    print("Charts saved to reports/figures/: uncertainty_decomposition.png, "
          "regime_pruning_comparison.png (if breaks found), reconciliation_comparison.png (if applicable)")

    return {
        "uncertainty_summary": uncertainty_summary,
        "regime_summary": regime_summary,
        "reconciliation_result": reconciliation_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "m5"], default="rossmann")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run_advanced_analysis(dataset=args.dataset, epochs=args.epochs)
