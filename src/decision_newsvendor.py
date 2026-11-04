
import numpy as np


def critical_ratio(holding_cost, stockout_cost):
    if holding_cost < 0 or stockout_cost < 0:
        raise ValueError("Costs cannot be negative")

    total = holding_cost + stockout_cost

    if total == 0:
        raise ValueError("At least one cost must be positive")

    return stockout_cost / total


def recommend_order_level(
    quantile_predictions,
    holding_cost,
    stockout_cost
):
    target_q = critical_ratio(
        holding_cost,
        stockout_cost
    )

    quantiles = np.asarray(
        sorted(quantile_predictions.keys()),
        dtype=float
    )

    values = np.asarray(
        [quantile_predictions[q] for q in quantiles],
        dtype=float
    )

    if len(quantiles) == 0:
        raise ValueError("No quantile predictions supplied")

    if np.any(np.diff(quantiles) <= 0):
        raise ValueError("Quantiles must be increasing")

    if not np.all(np.isfinite(values)):
        raise ValueError("Predictions must be finite")

    order = float(
        np.interp(target_q, quantiles, values)
    )

    closest = float(
        quantiles[np.argmin(np.abs(quantiles-target_q))]
    )

    return {
        "target_critical_ratio": target_q,
        "closest_available_quantile": closest,
        "recommended_order_level": max(0.0, order),
        "is_approximation": not np.isclose(
            target_q, closest
        ),
        "interpolated": not np.isclose(
            target_q, closest
        )
    }


def recommend_orders_by_series(
    predictions,
    holding_costs,
    stockout_costs
):
    """
    predictions:
        {sku_id: {0.1: forecast, 0.5: forecast, 0.9: forecast}}

    Costs can be individual mappings or common scalar values.
    """

    results = {}

    for sku, quantiles in predictions.items():

        h = (
            holding_costs[sku]
            if isinstance(holding_costs, dict)
            else holding_costs
        )

        s = (
            stockout_costs[sku]
            if isinstance(stockout_costs, dict)
            else stockout_costs
        )

        results[sku] = recommend_order_level(
            quantiles, h, s
        )

    return results


def simulate_cost(
    order_level,
    actual_demand,
    holding_cost,
    stockout_cost
):
    actual_demand = np.asarray(
        actual_demand, dtype=float
    )

    holding_units = np.maximum(
        order_level - actual_demand, 0
    )

    shortage_units = np.maximum(
        actual_demand - order_level, 0
    )

    holding = holding_units.sum() * holding_cost
    shortage = shortage_units.sum() * stockout_cost

    total = holding + shortage

    return {
        "total_holding_cost": float(holding),
        "total_shortage_cost": float(shortage),
        "total_cost": float(total),
        "avg_cost_per_period": float(
            total / max(1, len(actual_demand))
        ),
        "fill_rate": float(
            1 - shortage_units.sum()
            / max(actual_demand.sum(), 1)
        )
    }
