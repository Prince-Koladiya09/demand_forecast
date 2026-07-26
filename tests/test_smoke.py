"""
test_smoke.py
-------------
Basic smoke tests -- these don't validate scientific correctness of results,
just that each module's core functions run without error and return outputs
of the expected shape/type. Run with:  pytest tests/test_smoke.py -v
"""

import numpy as np
import torch

from src import dl_model
from src import calibration
from src import statistical_tests
from src import decision_newsvendor
from src import baseline_sarima
from src.experiment_log import log_experiment, load_experiment_log
from src.experiments.rolling_cv import rolling_origin_split
from src.experiments.segment_analysis import assign_volume_buckets


def test_quantile_lstm_forward_shape():
    batch_size, input_window, output_window = 4, 28, 7
    x = torch.randn(batch_size, input_window, 1)
    model = dl_model.QuantileLSTM(output_window=output_window)
    preds = model(x)
    assert preds.shape == (batch_size, 3, output_window)


def test_pinball_loss_is_nonnegative():
    preds = torch.randn(4, 3, 7)
    target = torch.randn(4, 7)
    loss = dl_model.pinball_loss(preds, target)
    assert loss.item() >= 0


def test_empirical_coverage_bounds():
    y_true = np.array([10, 20, 30, 40])
    lower = np.array([5, 15, 25, 45])   # last one should NOT be covered
    upper = np.array([15, 25, 35, 50])
    coverage = calibration.empirical_coverage(y_true, lower, upper)
    assert 0 <= coverage <= 1
    assert coverage == 0.75


def test_diebold_mariano_runs():
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 3, 100)
    e2 = rng.normal(0, 2, 100)
    result = statistical_tests.diebold_mariano_test(e1, e2)
    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1


def test_ljung_box_runs():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1, 100)
    result = statistical_tests.ljung_box_test(residuals)
    assert 0 <= result["p_value"] <= 1


def test_critical_ratio_bounds():
    ratio = decision_newsvendor.critical_ratio(holding_cost=1.0, stockout_cost=4.0)
    assert 0 < ratio < 1


def test_recommend_order_level_picks_closest_quantile():
    quantile_predictions = {0.1: 10, 0.5: 20, 0.9: 30}
    decision = decision_newsvendor.recommend_order_level(quantile_predictions, holding_cost=1.0, stockout_cost=4.0)
    assert decision["recommended_order_level"] in quantile_predictions.values()


def test_stationarity_check_runs():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 100))
    result = baseline_sarima.check_stationarity(series)
    assert "is_stationary" in result


def test_experiment_log_roundtrip(tmp_path):
    log_path = tmp_path / "test_log.csv"
    log_experiment({"run_type": "unit_test", "dataset": "synthetic", "tag": "t1", "pinball_loss": 1.0}, log_path=log_path)
    log_experiment({"run_type": "unit_test", "dataset": "synthetic", "tag": "t2", "pinball_loss": 2.0}, log_path=log_path)
    df = load_experiment_log(log_path)
    assert len(df) == 2
    assert set(df["tag"]) == {"t1", "t2"}


def test_rolling_origin_split_produces_expanding_folds():
    import pandas as pd
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    df = pd.concat([
        pd.DataFrame({"date": dates, "sku_id": sku, "units_sold": np.arange(200)})
        for sku in ["A", "B"]
    ], ignore_index=True)

    folds = rolling_origin_split(df, n_folds=3, test_frac_per_fold=0.1, min_train_frac=0.5)
    assert len(folds) == 3
    for train_df, test_df in folds:
        assert len(train_df) > 0
        assert len(test_df) > 0
    # training window should expand across folds
    assert len(folds[1][0]) > len(folds[0][0])
    assert len(folds[2][0]) > len(folds[1][0])


def test_assign_volume_buckets_creates_three_tiers():
    import pandas as pd
    rows = []
    for sku, base_volume in [("low_sku", 1), ("mid_sku", 10), ("high_sku", 100)]:
        for i in range(30):
            rows.append({"sku_id": sku, "units_sold": base_volume})
    df = pd.DataFrame(rows)

    buckets = assign_volume_buckets(df, n_buckets=3)
    assert buckets["low_sku"] == "low"
    assert buckets["high_sku"] == "high"
