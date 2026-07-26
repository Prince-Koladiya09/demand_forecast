"""
statistical_tests.py
---------------------
Implements FR-5: Statistical Model Comparison.

Two DIFFERENT questions are answered here, using two different tests:

1. Diebold-Mariano test: "Is the DL model's forecast error significantly
   lower than the baseline's, or could this just be noise?"
   -> Compares TWO models against each other.

2. Ljung-Box test: "Are this model's residuals just white noise, or is
   there leftover structure (e.g., weekly seasonality) it failed to learn?"
   -> Diagnoses ONE model on its own, independent of any comparison.

A plain paired t-test is NOT used for (1) because forecast errors over
time are typically autocorrelated, violating the independence assumption
a t-test requires. The Diebold-Mariano test is built specifically to
handle that.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def diebold_mariano_test(errors_model_1: np.ndarray, errors_model_2: np.ndarray, h: int = 1, power: int = 2):
    """
    Diebold-Mariano test comparing forecast accuracy of two models.

    errors_model_1, errors_model_2: arrays of forecast errors (y_true - y_pred)
        over the same test period, same length.
    h: forecast horizon (default 1-step; adjust if evaluating multi-step loss)
    power: 2 for squared-error loss differential (most common choice)

    Returns dict with DM statistic and two-sided p-value.
    Null hypothesis: both models have equal forecast accuracy.
    A significant p-value (< 0.05) means the difference is unlikely to be
    due to chance -- i.e., one model is genuinely better.
    """
    e1 = np.asarray(errors_model_1)
    e2 = np.asarray(errors_model_2)
    assert e1.shape == e2.shape, "Error series must be the same length"

    d = np.abs(e1) ** power - np.abs(e2) ** power  # loss differential
    n = len(d)
    d_mean = np.mean(d)

    # Newey-West style variance estimate accounting for autocorrelation
    # up to (h-1) lags, since forecast errors are typically autocorrelated.
    gamma_0 = np.var(d, ddof=0)
    var_d = gamma_0
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1] if lag < n else 0
        var_d += 2 * gamma_lag
    var_d /= n

    dm_stat = d_mean / np.sqrt(var_d) if var_d > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "model_1_better": d_mean < 0 and p_value < 0.05,
        "model_2_better": d_mean > 0 and p_value < 0.05,
    }


def ljung_box_test(residuals: np.ndarray, lags=10):
    """
    Ljung-Box test for autocorrelation in model residuals.

    Null hypothesis: residuals are independently distributed (white noise),
    i.e., the model has captured all the structure in the data.
    A significant p-value (< 0.05) at any tested lag means there is
    leftover autocorrelation -- the model is missing structure
    (e.g., weekly seasonality not fully captured).
    """
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = result["lb_pvalue"].iloc[0]
    return {
        "lb_statistic": result["lb_stat"].iloc[0],
        "p_value": p_value,
        "residuals_are_white_noise": p_value > 0.05,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate: model 2 (DL) has smaller errors than model 1 (baseline)
    errors_baseline = rng.normal(0, 3, 200)
    errors_dl = rng.normal(0, 2, 200)

    dm_result = diebold_mariano_test(errors_baseline, errors_dl)
    print("Diebold-Mariano test (baseline vs DL):", dm_result)

    # Simulate residuals: mostly white noise
    residuals = rng.normal(0, 1, 200)
    lb_result = ljung_box_test(residuals)
    print("Ljung-Box test:", lb_result)
