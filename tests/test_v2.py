
import numpy as np
import torch

from src.dl_model import QuantileLSTM
from src.decision_newsvendor import recommend_order_level
from src.metrics import pinball_loss
from src.inventory_simulation import simulate_order_up_to
from src.conformal import cqr_adjust


def test_quantiles_are_monotonic():

    model = QuantileLSTM(
        input_size=2,
        output_window=7
    )

    x = torch.randn(8, 28, 2)

    predictions = model(x)

    assert torch.all(
        predictions[:, 1:] >= predictions[:, :-1]
    )


def test_newsvendor_interpolation():

    predictions = {
        0.1: 10,
        0.5: 20,
        0.9: 30
    }

    result = recommend_order_level(
        predictions,
        holding_cost=1,
        stockout_cost=4
    )

    assert np.isclose(
        result["recommended_order_level"],
        27.5
    )


def test_pinball_loss():

    loss = pinball_loss(
        [10, 10],
        [8, 12],
        0.5
    )

    assert np.isclose(loss, 1.0)


def test_conformal_calibration():

    result = cqr_adjust(
        np.zeros(20),
        np.ones(20),
        np.full(20, 2.0)
    )

    assert np.all(result["lower"] <= 0)
    assert np.all(result["upper"] >= 1)


def test_inventory_simulation():

    result = simulate_order_up_to(
        demand=[5, 10, 8],
        forecasts=[8, 12, 10],
        lead_time=0
    )

    assert result["total_cost"] >= 0
    assert 0 <= result["fill_rate"] <= 1
