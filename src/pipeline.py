"""
pipeline.py
-----------
Orchestrates the full project end-to-end, matching the flow in the SRS
(Section 5 - System Architecture):

Raw Data -> Preprocessing -> SARIMA Baseline + LSTM Quantile Model
         -> Calibration Check -> Statistical Comparison (DM, Ljung-Box)
         -> Newsvendor Decision -> Cost Simulation Report

Run with:  python -m src.pipeline

NOTE: This script assumes a real dataset has been placed in data/raw/
and that src/preprocessing.py's column names have been adjusted to match
it. Where a real dataset isn't available yet, each module's own
`if __name__ == "__main__":` block can be run individually with synthetic
data to verify correctness (see each file's smoke test).
"""

import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src import preprocessing
from src import baseline_sarima
from src import dl_model
from src import calibration
from src import statistical_tests
from src import decision_newsvendor
from src.experiment_log import log_experiment


def load_dataset(name: str):
    """
    name: "rossmann", "m5", or "generic"
    Returns (df, feature_cols):
        df: standardized long-format DataFrame (date, sku_id, units_sold, ...extra feature columns)
            ready to be passed into preprocessing.run_preprocessing_pipeline(df=...).
        feature_cols: list of extra numeric column names in `df` to use as
            additional input channels (empty list if none).
    "generic" instead reads directly from preprocessing.RAW_DATA_PATH and
    uses no extra features.
    """
    if name == "rossmann":
        from src.data_loaders.rossmann_loader import load_rossmann, FEATURE_COLS
        return load_rossmann(), FEATURE_COLS
    elif name == "m5":
        from src.data_loaders.m5_loader import load_m5, FEATURE_COLS
        return load_m5(n_series=300), FEATURE_COLS
    elif name == "generic":
        return None, []  # preprocessing.run_preprocessing_pipeline will read RAW_DATA_PATH directly
    else:
        raise ValueError(f"Unknown dataset name: {name}. Choose 'rossmann', 'm5', or 'generic'.")


def build_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    X_train/X_val already have shape (n_samples, input_window, n_channels)
    from src/preprocessing.py's make_windows() -- channel 0 is always the
    target (units_sold), followed by any extra feature channels. No extra
    dimension needs to be added here.
    """
    def to_tensor(X, y):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor(X_train, y_train)
    val_ds = to_tensor(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_full_pipeline(dataset: str = "rossmann"):
    print(f"=== Step 1: Preprocessing ({dataset}) ===")
    df, feature_cols = load_dataset(dataset)
    data = preprocessing.run_preprocessing_pipeline(df=df, feature_cols=feature_cols, save_as=dataset)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, sku_test = data["test"]
    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("Extra feature channels used:", feature_cols)

    print("\n=== Step 2: SARIMA Baseline (per first SKU test window, as example) ===")
    # Channel 0 is always the target (units_sold) -- see make_windows() docstring.
    first_series = X_test[0][:, 0]
    stat_check = baseline_sarima.check_stationarity(first_series)
    print("Stationarity check:", stat_check)
    sarima_fitted = baseline_sarima.fit_sarima(first_series)
    sarima_forecast = baseline_sarima.forecast_sarima(sarima_fitted, steps=y_test.shape[1])
    print("SARIMA forecast (mean):", sarima_forecast["mean"])

    print("\n=== Step 3: Train LSTM Quantile Model ===")
    train_loader, val_loader = build_dataloaders(X_train, y_train, X_val, y_val)
    n_channels = X_train.shape[-1]  # 1 (target only) or more if extra features are used
    model = dl_model.QuantileLSTM(input_size=n_channels, output_window=y_train.shape[1])
    model = dl_model.train_model(model, train_loader, val_loader, epochs=10)

    print("\n=== Step 4: Predict on Test Set + Calibration Check ===")
    preds = dl_model.predict(model, X_test)  # (n, 3, output_window)
    p10, p50, p90 = preds[:, 0, :], preds[:, 1, :], preds[:, 2, :]

    coverage = calibration.empirical_coverage(y_test, p10, p90)
    calib_error = calibration.calibration_error(y_test, p10, p90)
    print(f"Empirical coverage (target 80%): {coverage:.3f}")
    print(f"Calibration error: {calib_error:.3f}")

    print("\n=== Step 5: Statistical Comparison ===")
    dl_errors = (y_test - p50).flatten()
    baseline_errors = (y_test[0] - sarima_forecast["mean"]).flatten()  # example series only
    # NOTE: for a fair full-dataset DM test, compute baseline errors across
    # the same test set as the DL model, not just one example series.
    if len(baseline_errors) == len(dl_errors[:len(baseline_errors)]):
        dm_result = statistical_tests.diebold_mariano_test(
            baseline_errors, dl_errors[:len(baseline_errors)]
        )
        print("Diebold-Mariano result:", dm_result)

    lb_result = statistical_tests.ljung_box_test(dl_errors[:50])
    print("Ljung-Box result (DL residuals):", lb_result)

    print("\n=== Step 6: Newsvendor Decision + Cost Simulation ===")
    holding_cost, stockout_cost = 1.0, 4.0
    example_quantiles = {0.1: float(np.mean(p10)), 0.5: float(np.mean(p50)), 0.9: float(np.mean(p90))}
    decision = decision_newsvendor.recommend_order_level(example_quantiles, holding_cost, stockout_cost)
    print("Decision:", decision)

    cost_result = decision_newsvendor.simulate_cost(
        decision["recommended_order_level"], y_test.flatten(), holding_cost, stockout_cost
    )
    print("Cost simulation result:", cost_result)

    print("\n=== Step 7: Logging Experiment Result ===")
    log_experiment({
        "run_type": "main_pipeline", "dataset": dataset, "tag": "full_run",
        "hidden_size": 64, "input_window": X_train.shape[1], "output_window": y_train.shape[1],
        "n_features": n_channels, "feature_cols": str(feature_cols), "epochs": 10,
        "pinball_loss": round(float(dl_model.pinball_loss(
            torch.as_tensor(preds, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.float32)
        ).item()), 4),
        "mae": round(float(np.mean(np.abs(y_test - p50))), 4),
        "empirical_coverage": round(coverage, 4),
        "calibration_error": round(calib_error, 4),
        "dm_p_value": round(dm_result["p_value"], 4) if 'dm_result' in locals() else "",
        "ljung_box_p_value": round(lb_result["p_value"], 4),
        "notes": f"total_cost={cost_result['total_cost']:.2f}",
    })
    print("Result appended to reports/experiment_log.csv")

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["rossmann", "m5", "generic"], default="rossmann",
        help="Which dataset to run the pipeline on."
    )
    args = parser.parse_args()
    run_full_pipeline(dataset=args.dataset)
