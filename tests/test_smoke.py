
"""
test_smoke.py
-------------
Basic smoke tests for the demand forecasting project.

These tests validate that core functions execute correctly
and return outputs of the expected shape and type.

Run:
    python -m pytest tests/test_smoke.py -v
"""

import numpy as np
import torch

from src import dl_model
from src import calibration
from src import statistical_tests
from src import decision_newsvendor
from src import baseline_sarima

from src.experiment_log import (
    log_experiment,
    load_experiment_log
)

from src.experiments.rolling_cv import rolling_origin_split
from src.experiments.segment_analysis import assign_volume_buckets

from src.uncertainty_decomposition import (
    mc_dropout_predict,
    summarize_by_horizon
)

from src.structural_break_detection import (
    cusum_break_detection,
    detect_breaks_per_sku,
    apply_regime_pruning
)

from src.hierarchical_reconciliation import (
    reconcile_forecasts,
    reconcile_forecast_matrix
)


def test_quantile_lstm_forward_shape():

    batch_size = 4
    input_window = 28
    output_window = 7

    x = torch.randn(batch_size, input_window, 1)

    model = dl_model.QuantileLSTM(
        output_window=output_window
    )

    preds = model(x)

    assert preds.shape == (
        batch_size, 3, output_window
    )


def test_pinball_loss_is_nonnegative():

    preds = torch.randn(4, 3, 7)
    target = torch.randn(4, 7)

    loss = dl_model.pinball_loss(preds, target)

    assert loss.item() >= 0


def test_empirical_coverage_bounds():

    y_true = np.array([10, 20, 30, 40])

    lower = np.array([5, 15, 25, 45])
    upper = np.array([15, 25, 35, 50])

    coverage = calibration.empirical_coverage(
        y_true, lower, upper
    )

    assert 0 <= coverage <= 1
    assert coverage == 0.75


def test_diebold_mariano_runs():

    rng = np.random.default_rng(0)

    e1 = rng.normal(0, 3, 100)
    e2 = rng.normal(0, 2, 100)

    result = statistical_tests.diebold_mariano_test(
        e1, e2
    )

    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1
    assert "dm_statistic" in result


def test_ljung_box_runs():

    rng = np.random.default_rng(0)

    residuals = rng.normal(0, 1, 100)

    result = statistical_tests.ljung_box_test(
        residuals
    )

    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1
    assert "lb_statistic" in result


def test_critical_ratio_bounds():

    ratio = decision_newsvendor.critical_ratio(
        holding_cost=1.0,
        stockout_cost=4.0
    )

    assert 0 < ratio < 1

    assert np.isclose(ratio, 0.8)


def test_recommend_order_level_interpolates_quantiles():

    quantile_predictions = {
        0.1: 10,
        0.5: 20,
        0.9: 30
    }

    decision = decision_newsvendor.recommend_order_level(
        quantile_predictions,
        holding_cost=1.0,
        stockout_cost=4.0
    )

    # Critical ratio = 4 / (4 + 1) = 0.80
    assert np.isclose(
        decision["target_critical_ratio"],
        0.8
    )

    # Interpolation between P50=20 and P90=30
    assert np.isclose(
        decision["recommended_order_level"],
        27.5
    )

    assert decision["interpolated"] is True

    assert decision["recommended_order_level"] >= 0


def test_stationarity_check_runs():

    rng = np.random.default_rng(0)

    series = np.cumsum(
        rng.normal(0, 1, 100)
    )

    result = baseline_sarima.check_stationarity(
        series
    )

    assert "is_stationary" in result


def test_experiment_log_roundtrip(tmp_path):

    log_path = tmp_path / "test_log.csv"

    log_experiment(
        {
            "run_type": "unit_test",
            "dataset": "synthetic",
            "tag": "t1",
            "pinball_loss": 1.0
        },
        log_path=log_path
    )

    log_experiment(
        {
            "run_type": "unit_test",
            "dataset": "synthetic",
            "tag": "t2",
            "pinball_loss": 2.0
        },
        log_path=log_path
    )

    df = load_experiment_log(log_path)

    assert len(df) == 2
    assert set(df["tag"]) == {"t1", "t2"}


def test_rolling_origin_split_produces_expanding_folds():

    import pandas as pd

    dates = pd.date_range(
        "2021-01-01",
        periods=200,
        freq="D"
    )

    df = pd.concat(
        [
            pd.DataFrame({
                "date": dates,
                "sku_id": sku,
                "units_sold": np.arange(200)
            })
            for sku in ["A", "B"]
        ],
        ignore_index=True
    )

    folds = rolling_origin_split(
        df,
        n_folds=3,
        test_frac_per_fold=0.1,
        min_train_frac=0.5
    )

    assert len(folds) == 3

    for train_df, test_df in folds:

        assert len(train_df) > 0
        assert len(test_df) > 0

    # Training window should expand across folds.
    assert len(folds[1][0]) > len(folds[0][0])
    assert len(folds[2][0]) > len(folds[1][0])


def test_assign_volume_buckets_creates_three_tiers():

    import pandas as pd

    rows = []

    for sku, base_volume in [
        ("low_sku", 1),
        ("mid_sku", 10),
        ("high_sku", 100)
    ]:

        for i in range(30):

            rows.append({
                "sku_id": sku,
                "units_sold": base_volume
            })

    df = pd.DataFrame(rows)

    buckets = assign_volume_buckets(
        df,
        n_buckets=3
    )

    assert buckets["low_sku"] == "low"
    assert buckets["high_sku"] == "high"


def test_mc_dropout_predict_shapes():

    batch_size = 10
    input_window = 28
    output_window = 7

    dummy_X = np.random.randn(
        batch_size,
        input_window,
        1
    ).astype(np.float32)

    model = dl_model.QuantileLSTM(
        output_window=output_window,
        dropout=0.3
    )

    mean_preds, epistemic_std, aleatoric_spread = (
        mc_dropout_predict(
            model,
            dummy_X,
            n_passes=5
        )
    )

    assert mean_preds.shape == (
        batch_size, 3, output_window
    )

    assert epistemic_std.shape == (
        batch_size, output_window
    )

    assert aleatoric_spread.shape == (
        batch_size, output_window
    )

    assert np.all(aleatoric_spread >= 0)

    summary = summarize_by_horizon(
        epistemic_std,
        aleatoric_spread
    )

    assert summary["epistemic_by_horizon"].shape == (
        output_window,
    )

    assert summary["aleatoric_by_horizon"].shape == (
        output_window,
    )


def test_cusum_detects_injected_break_with_no_false_positive():

    rng = np.random.default_rng(0)

    before = rng.normal(20, 3, 100)
    after = rng.normal(50, 3, 100)

    series_with_break = np.concatenate(
        [before, after]
    )

    break_idx = cusum_break_detection(
        series_with_break
    )

    assert break_idx is not None

    assert abs(break_idx - 100) <= 5

    # No false positive on pure noise.
    stationary_series = rng.normal(20, 3, 200)

    assert cusum_break_detection(
        stationary_series
    ) is None


def test_regime_pruning_removes_only_pre_break_rows():

    import pandas as pd

    dates = pd.date_range(
        "2021-01-01",
        periods=200,
        freq="D"
    )

    rng = np.random.default_rng(0)

    df = pd.DataFrame({
        "date": dates,
        "sku_id": "A",
        "units_sold": np.concatenate([
            rng.normal(20, 3, 100),
            rng.normal(50, 3, 100)
        ])
    })

    breaks = detect_breaks_per_sku(
        df,
        sku_col="sku_id",
        date_col="date",
        target_col="units_sold"
    )

    pruned = apply_regime_pruning(
        df,
        sku_col="sku_id",
        date_col="date",
        breaks=breaks
    )

    assert len(pruned) < len(df)

    assert (
        pruned["units_sold"].mean()
        > df["units_sold"].mean()
    )


def test_reconcile_forecasts_sums_consistently():

    bottom_forecasts = np.array([
        20.0, 35.0, 15.0, 28.0
    ])

    bottom_variances = np.array([
        4.0, 9.0, 2.0, 6.0
    ])

    result = reconcile_forecasts(
        bottom_forecasts,
        bottom_variances,
        top_forecast=110.0,
        top_variance=5.0
    )

    assert np.isclose(
        result["bottom_reconciled"].sum(),
        result["top_reconciled"]
    )

    assert (
        result["top_raw"]
        < result["top_reconciled"]
        < 110.0 + 1e-6
    )


def test_reconcile_forecast_matrix_shapes():

    forecast_matrix = np.array([
        [20.0, 21.0],
        [35.0, 34.0],
        [15.0, 16.0]
    ])

    bottom_variances = np.array([
        4.0, 9.0, 2.0
    ])

    top_series = np.array([
        80.0, 78.0
    ])

    result = reconcile_forecast_matrix(
        forecast_matrix,
        bottom_variances,
        top_forecast_series=top_series
    )

    assert result["bottom_reconciled"].shape == (
        forecast_matrix.shape
    )

    assert result["top_reconciled"].shape == (2,)

    for h in range(2):

        assert np.isclose(
            result["bottom_reconciled"][:, h].sum(),
            result["top_reconciled"][h]
        )
