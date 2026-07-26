"""
baseline_sarima.py
-------------------
Implements FR-2: Baseline Statistical Model.

Fits a SARIMA model per SKU as a classical baseline, after checking
stationarity with the Augmented Dickey-Fuller (ADF) test and differencing
if needed. Produces point forecasts + a Gaussian confidence interval,
used purely as a comparison point for the DL model.
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run the Augmented Dickey-Fuller test.
    Null hypothesis: the series has a unit root (i.e., is non-stationary).
    If p-value < alpha, we reject the null -> series is stationary.
    """
    result = adfuller(series)
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < alpha,
    }


def fit_sarima(series: np.ndarray, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Fit a SARIMA model. Default seasonal_order assumes weekly seasonality
    (period=7), which is standard for daily retail demand data.

    If check_stationarity() indicates non-stationarity, the (1,1,1) order's
    middle "1" (differencing term) already handles first-order differencing,
    so no separate manual differencing step is required.
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def forecast_sarima(fitted_model, steps: int, alpha: float = 0.20):
    """
    Produce point forecast + confidence interval.
    alpha=0.20 -> 80% confidence interval, to match the P10/P90 band
    used by the DL model for a fair comparison.
    """
    forecast_result = fitted_model.get_forecast(steps=steps)
    mean_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)
    return {
        "mean": np.asarray(mean_forecast),
        "lower": np.asarray(conf_int[:, 0]),
        "upper": np.asarray(conf_int[:, 1]),
    }


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    rng = np.random.default_rng(42)
    synthetic_series = np.cumsum(rng.normal(0, 1, 200)) + 50

    stat_check = check_stationarity(synthetic_series)
    print("Stationarity check:", stat_check)

    fitted = fit_sarima(synthetic_series)
    forecast = forecast_sarima(fitted, steps=7)
    print("7-day forecast (mean):", forecast["mean"])
    print("80% interval lower:  ", forecast["lower"])
    print("80% interval upper:  ", forecast["upper"])
