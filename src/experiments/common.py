"""
experiments/common.py
-----------------------
Shared training/evaluation routine used by:
    - hyperparam_search.py
    - ablation_study.py
    - rolling_cv.py

Centralizing this avoids each experiment script re-implementing its own
slightly-different training loop, which would make results harder to
compare fairly across experiments (a classic bug: comparing two configs
that were secretly evaluated differently).
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src import preprocessing
from src import dl_model
from src import calibration
from src.experiment_log import log_experiment


def build_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    def to_tensor(X, y):
        return TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def train_and_evaluate(
    train_df=None, val_df=None, test_df=None,
    X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
    feature_cols=None, hidden_size=64, input_window=28, output_window=7,
    epochs=8, run_type="experiment", dataset_name="unknown", tag="", fold=None,
    verbose=True,
):
    """
    Two ways to call this:
    1. Pass train_df/val_df/test_df (long-format DataFrames, e.g. from a
       rolling-CV fold split) -- windows are built internally.
    2. Pass already-windowed X_train/y_train/etc. directly (e.g. from
       preprocessing.run_preprocessing_pipeline()) -- skips re-windowing.

    Trains a QuantileLSTM, evaluates pinball loss / MAE / calibration on
    the test set, logs the result via experiment_log.log_experiment(), and
    returns the metrics dict.
    """
    feature_cols = feature_cols or []

    if X_train is None:
        X_train, y_train, _ = preprocessing.make_windows(train_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)
        X_val, y_val, _ = preprocessing.make_windows(val_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)
        X_test, y_test, sku_test = preprocessing.make_windows(test_df, input_window=input_window, output_window=output_window, feature_cols=feature_cols)
    else:
        sku_test = None

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Not enough history to build windows for input_window={input_window}, "
            f"output_window={output_window} with this data. Try a shorter input_window "
            f"or a dataset with more history per series."
        )

    train_loader, val_loader = build_dataloaders(X_train, y_train, X_val, y_val)
    n_channels = X_train.shape[-1]

    model = dl_model.QuantileLSTM(input_size=n_channels, hidden_size=hidden_size, output_window=output_window)
    model = dl_model.train_model(model, train_loader, val_loader, epochs=epochs)

    preds = dl_model.predict(model, X_test)  # (n, 3, output_window)
    p10, p50, p90 = preds[:, 0, :], preds[:, 1, :], preds[:, 2, :]

    pinball = float(dl_model.pinball_loss(
        torch.as_tensor(preds, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.float32),
    ).item())
    mae_val = mae(y_test, p50)
    coverage = calibration.empirical_coverage(y_test, p10, p90)
    calib_err = calibration.calibration_error(y_test, p10, p90)

    metrics = {
        "run_type": run_type, "dataset": dataset_name, "tag": tag,
        "hidden_size": hidden_size, "input_window": input_window, "output_window": output_window,
        "n_features": n_channels, "feature_cols": str(feature_cols), "epochs": epochs, "fold": fold,
        "pinball_loss": round(pinball, 4), "mae": round(mae_val, 4),
        "empirical_coverage": round(coverage, 4), "calibration_error": round(calib_err, 4),
    }

    log_experiment(metrics)

    if verbose:
        print(f"[{run_type}/{tag}] pinball={pinball:.4f} mae={mae_val:.4f} coverage={coverage:.3f} calib_err={calib_err:.3f}")

    return {**metrics, "model": model, "preds": preds, "y_test": y_test, "sku_test": sku_test,
            "p10": p10, "p50": p50, "p90": p90}
