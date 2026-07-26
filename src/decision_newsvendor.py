"""
decision_newsvendor.py
------------------------
Implements FR-6: Inventory Decision Layer.

The classical newsvendor problem: given a holding cost (h) for unsold
stock and a shortage/stockout cost (s) for unmet demand, the optimal
order quantity is the quantile of the demand distribution corresponding
to the "critical ratio" s / (s + h).

Because our DL model already outputs quantiles (P10/P50/P90) directly,
this becomes a simple lookup -- no distribution needs to be assumed,
and no optimizer is required. This is the payoff of using quantile
regression instead of a point forecast: the model's output is directly
usable in a real business decision.
"""

import numpy as np


def critical_ratio(holding_cost: float, stockout_cost: float) -> float:
    """
    Critical ratio = s / (s + h)
    This is the quantile of the demand distribution at which expected
    holding cost of one extra unit equals expected stockout cost of
    one fewer unit -- i.e., the profit-maximizing order level.
    """
    return stockout_cost / (stockout_cost + holding_cost)


def recommend_order_level(quantile_predictions: dict, holding_cost: float, stockout_cost: float):
    """
    quantile_predictions: dict mapping quantile (e.g. 0.1, 0.5, 0.9) -> predicted demand value

    If the critical ratio doesn't exactly match one of the available
    quantiles, this picks the closest available quantile (since our model
    only outputs P10/P50/P90) and flags that an approximation was made.
    """
    target_q = critical_ratio(holding_cost, stockout_cost)
    available_qs = np.array(list(quantile_predictions.keys()))
    closest_q = available_qs[np.argmin(np.abs(available_qs - target_q))]

    return {
        "target_critical_ratio": target_q,
        "closest_available_quantile": closest_q,
        "recommended_order_level": quantile_predictions[closest_q],
        "is_approximation": not np.isclose(target_q, closest_q, atol=1e-6),
    }


def simulate_cost(order_level: float, actual_demand: np.ndarray, holding_cost: float, stockout_cost: float) -> dict:
    """
    Simulate total cost of using `order_level` as the fixed order quantity
    against a series of actual demand realizations.

    holding units = max(order_level - demand, 0)  -> cost = holding_cost * that
    shortage units = max(demand - order_level, 0)  -> cost = stockout_cost * that
    """
    actual_demand = np.asarray(actual_demand)
    holding_units = np.maximum(order_level - actual_demand, 0)
    shortage_units = np.maximum(actual_demand - order_level, 0)

    total_holding_cost = np.sum(holding_units) * holding_cost
    total_shortage_cost = np.sum(shortage_units) * stockout_cost
    total_cost = total_holding_cost + total_shortage_cost

    return {
        "total_holding_cost": total_holding_cost,
        "total_shortage_cost": total_shortage_cost,
        "total_cost": total_cost,
        "avg_cost_per_period": total_cost / len(actual_demand),
    }


if __name__ == "__main__":
    # Minimal smoke test
    holding_cost = 1.0     # cost per unit of unsold inventory
    stockout_cost = 4.0    # cost per unit of unmet demand (usually higher than holding cost)

    quantile_predictions = {0.1: 12, 0.5: 20, 0.9: 30}

    decision = recommend_order_level(quantile_predictions, holding_cost, stockout_cost)
    print("Decision:", decision)

    rng = np.random.default_rng(1)
    simulated_demand = rng.poisson(20, size=30)
    cost_result = simulate_cost(decision["recommended_order_level"], simulated_demand, holding_cost, stockout_cost)
    print("Cost simulation:", cost_result)
