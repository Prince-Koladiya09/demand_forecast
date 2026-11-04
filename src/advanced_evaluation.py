
import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def seasonal_naive(history, horizon=7, season_length=7):
    """
    Repeats the last observed seasonal cycle.
    """

    history = np.asarray(history, dtype=float)

    if len(history) < season_length:
        raise ValueError("Insufficient history")

    return np.resize(
        history[-season_length:],
        horizon
    )


def ets_forecast(history, horizon=7, season_length=7):
    """
    Exponential smoothing baseline.
    """

    history = np.asarray(history, dtype=float)

    if len(history) < 2 * season_length:
        return seasonal_naive(
            history,
            horizon,
            season_length
        )

    model = ExponentialSmoothing(
        history,
        trend="add",
        seasonal="add",
        seasonal_periods=season_length,
        initialization_method="estimated"
    )

    fitted = model.fit(optimized=True)

    return np.asarray(
        fitted.forecast(horizon)
    )


def fixed_forecast_origins(
    frame,
    date_col="date",
    series_col="sku_id",
    n_origins=4,
    horizon=7
):
    """
    Generates common forecast origins.

    The same origins should be supplied to every
    forecasting model being compared.
    """

    dates = pd.Index(
        sorted(pd.to_datetime(frame[date_col].unique()))
    )

    origins = dates[-(n_origins * horizon):][::horizon]
    origins = origins[-n_origins:]

    output = []

    for sku, group in frame.groupby(series_col):

        available_dates = set(
            pd.to_datetime(group[date_col])
        )

        for origin in origins:

            target_dates = pd.date_range(
                origin,
                periods=horizon,
                freq="D"
            )

            if all(d in available_dates for d in target_dates):

                output.append({
                    "sku_id": sku,
                    "origin": origin,
                    "target_dates": target_dates
                })

    return output


def summarize_seeds(results):
    """
    Aggregates metrics over multiple random seeds.
    """

    df = pd.DataFrame(results)

    numeric = df.select_dtypes(include="number")

    return pd.DataFrame({
        "mean": numeric.mean(),
        "std": numeric.std(ddof=1)
    })


def paired_bootstrap_difference(
    cost_a,
    cost_b,
    n_boot=2000,
    seed=42
):
    """
    Paired bootstrap CI for cost A minus cost B.
    Negative values favor policy A.
    """

    a = np.asarray(cost_a, dtype=float)
    b = np.asarray(cost_b, dtype=float)

    if a.shape != b.shape or len(a) == 0:
        raise ValueError("Arrays must be paired and non-empty")

    difference = a - b

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        difference,
        size=(n_boot, len(difference)),
        replace=True
    ).mean(axis=1)

    return {
        "mean_difference": float(difference.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975))
    }
