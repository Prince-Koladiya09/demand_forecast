"""
experiments/ablation_study.py
--------------------------------
FR: Feature importance / ablation study.

Answers a specific question: do the extra feature channels we added
(sell_price for M5; promo/competition_distance/promo2/store_type_code/
assortment_code for Rossmann) actually improve forecasts, or were they
added without evidence they help?

Method: train the identical model (same hidden_size, input_window, epochs)
twice -- once with only the target channel, once with the full feature
set -- and compare pinball loss and calibration on the same test set.
This isolates the effect of the features from any other change, which is
why it's a fair ablation and not just "two different runs."

Usage:
    python -m src.experiments.ablation_study --dataset rossmann
"""

import argparse
import pandas as pd

from src.pipeline import load_dataset
from src import preprocessing
from src.experiments.common import train_and_evaluate
from src.experiments.plots import plot_ablation_comparison


def run_ablation_study(dataset: str, hidden_size: int = 64, input_window: int = 28, epochs: int = 8):
    print(f"=== Feature Ablation Study on '{dataset}' ===")
    df, full_feature_cols = load_dataset(dataset)

    configs = {
        "target_only": [],
        "with_extra_features": full_feature_cols,
    }

    results = []
    for tag, feature_cols in configs.items():
        print(f"\n--- Config: {tag} (features={feature_cols}) ---")
        data = preprocessing.run_preprocessing_pipeline(
            df=df, feature_cols=feature_cols, input_window=input_window,
        )
        X_train, y_train, _ = data["train"]
        X_val, y_val, _ = data["val"]
        X_test, y_test, _ = data["test"]

        result = train_and_evaluate(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_cols=feature_cols, hidden_size=hidden_size, input_window=input_window,
            epochs=epochs, run_type="ablation_study", dataset_name=dataset, tag=tag,
        )
        results.append(result)

    summary = pd.DataFrame([
        {"config": tag, "n_features": r["n_features"], "pinball_loss": r["pinball_loss"],
         "mae": r["mae"], "empirical_coverage": r["empirical_coverage"],
         "calibration_error": r["calibration_error"]}
        for tag, r in zip(configs.keys(), results)
    ])

    print("\n=== Ablation Results ===")
    print(summary.to_string(index=False))

    plot_ablation_comparison(summary)

    baseline_pinball = summary.loc[summary["config"] == "target_only", "pinball_loss"].iloc[0]
    with_features_pinball = summary.loc[summary["config"] == "with_extra_features", "pinball_loss"].iloc[0]
    pct_change = 100 * (with_features_pinball - baseline_pinball) / baseline_pinball

    if with_features_pinball < baseline_pinball:
        print(f"\nExtra features IMPROVED pinball loss by {abs(pct_change):.1f}% -- worth keeping.")
    else:
        print(f"\nExtra features did NOT improve pinball loss (change: {pct_change:+.1f}%). "
              f"On this data/config, the added features are not clearly earning their keep -- "
              f"worth investigating further before assuming they help.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rossmann", "m5"], default="rossmann")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    run_ablation_study(dataset=args.dataset, epochs=args.epochs)
