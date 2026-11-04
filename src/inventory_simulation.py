
import numpy as np


def simulate_order_up_to(
    demand,
    forecasts,
    lead_time=1,
    holding_cost=1.0,
    stockout_cost=4.0,
    initial_inventory=0.0
):
    """
    Simulates an order-up-to inventory policy.

    Orders are based on forecasts available at each review period.
    """

    demand = np.asarray(demand, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)

    if demand.shape != forecasts.shape:
        raise ValueError("Demand and forecasts must match")

    if lead_time < 0:
        raise ValueError("Lead time cannot be negative")

    if holding_cost < 0 or stockout_cost < 0:
        raise ValueError("Costs cannot be negative")

    on_hand = float(initial_inventory)

    pipeline = [0.0] * (lead_time + 1)

    total_holding = 0.0
    total_stockout = 0.0
    records = []

    for t, actual in enumerate(demand):

        # Receive scheduled inventory
        on_hand += pipeline.pop(0)
        pipeline.append(0.0)

        sold = min(on_hand, actual)
        shortage = max(0.0, actual - sold)

        on_hand -= sold

        holding = on_hand * holding_cost
        stockout = shortage * stockout_cost

        total_holding += holding
        total_stockout += stockout

        inventory_position = on_hand + sum(pipeline)

        order = max(
            0.0,
            forecasts[t] - inventory_position
        )

        if lead_time == 0:
            on_hand += order
        else:
            pipeline[-1] += order

        records.append({
            "period": t,
            "demand": float(actual),
            "sold": float(sold),
            "stockout_units": float(shortage),
            "ending_inventory": float(on_hand),
            "order_quantity": float(order),
            "holding_cost": float(holding),
            "stockout_cost": float(stockout)
        })

    total_cost = total_holding + total_stockout

    total_demand = max(float(demand.sum()), 1.0)

    fill_rate = (
        1 - sum(r["stockout_units"] for r in records)
        / total_demand
    )

    return {
        "total_cost": float(total_cost),
        "holding_cost": float(total_holding),
        "stockout_cost": float(total_stockout),
        "fill_rate": float(fill_rate),
        "average_inventory": float(
            np.mean([
                r["ending_inventory"] for r in records
            ])
        ),
        "periods": records
    }
