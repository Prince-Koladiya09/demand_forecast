"""
hierarchical_reconciliation.py
---------------------------------
Reconciles forecasts across a hierarchy (e.g. individual SKU/store forecasts
and their total) so that the reconciled bottom-level forecasts sum EXACTLY
to the reconciled top-level forecast. Raw independent forecasts from
per-series models (like our LSTM, run separately per SKU) almost never
satisfy this automatically -- e.g. 5 stores' forecasts might sum to 214,
while a separate total-level forecast says 220. Reconciliation removes that
inconsistency, which matters because business stakeholders (e.g. regional
vs. store-level planners) need numbers that agree with each other.

IMPORTANT IMPLEMENTATION NOTE: there is no `statsmodels.tsa.forecasting.
reconciliation` module -- that API does not exist in statsmodels as of this
project's dependencies. This module implements the standard hierarchical
reconciliation formula directly with numpy, which is straightforward and
avoids depending on a library function that isn't actually available.

Method: "structural scaling" reconciliation (a simplified, diagonal-weight
variant of MinT -- Minimum Trace reconciliation, Wickramasuriya et al. 2019).
Full MinT uses the complete forecast-error covariance matrix between all
series, which requires substantial in-sample residual history to estimate
reliably. Structural scaling instead uses a diagonal approximation (each
series' own variance, ignoring cross-series covariance) -- less statistically
optimal than full MinT, but much more stable to estimate with limited data,
and still guarantees hierarchical consistency (the actual business
requirement). This tradeoff is explicit and appropriate for this project's
data scale, not an oversight.

The reconciliation formula used, given:
    S: summing matrix (how bottom series aggregate to the top)
    y_hat: stacked vector of [top_forecast, bottom_forecasts...]
    W: diagonal weight matrix of each series' forecast error variance

    y_reconciled = S (S' W^-1 S)^-1 S' W^-1 y_hat

This is applied independently at each forecast horizon step (t+1, t+2, ...),
since each horizon step's forecasts are a separate reconciliation problem.
"""

import numpy as np


def build_summing_matrix(n_bottom: int) -> np.ndarray:
    """
    For a simple two-level hierarchy (one total + n_bottom individual
    series), the summing matrix S has shape (n_bottom + 1, n_bottom):
        - First row: all 1s (top = sum of all bottom series)
        - Remaining rows: identity (each bottom series maps to itself)
    """
    top_row = np.ones((1, n_bottom))
    bottom_rows = np.eye(n_bottom)
    return np.vstack([top_row, bottom_rows])  # shape (n_bottom + 1, n_bottom)


def reconcile_forecasts(bottom_forecasts: np.ndarray, bottom_variances: np.ndarray = None,
                         top_forecast: float = None, top_variance: float = None) -> dict:
    """
    bottom_forecasts: shape (n_series,) -- one forecast value per series at
        a SINGLE horizon step (call this once per horizon step you want
        reconciled).
    bottom_variances: shape (n_series,) -- optional per-series forecast
        error variance (e.g. from historical residuals). If not given,
        equal weighting is used (equivalent to simple "bottom-up plus OLS"
        reconciliation rather than variance-weighted MinT-style).
    top_forecast: an INDEPENDENTLY produced top-level (aggregate) forecast
        -- e.g. from a separate model trained directly on total demand,
        rather than just summing the bottom-level forecasts. This is the
        realistic scenario reconciliation solves: the top and bottom
        forecasts came from different models and disagree with each other.
        If not given, falls back to the bottom-up sum (in which case
        reconciliation has nothing to correct, since there's only one
        source of information -- this fallback exists for convenience,
        not as the typical use case).
    top_variance: forecast error variance of the independent top-level
        forecast. If not given, defaults to the sum of bottom variances
        (the structural-scaling assumption).

    Returns a dict with:
        - "top_reconciled": reconciled top-level (total) forecast
        - "bottom_reconciled": reconciled per-series forecasts, shape (n_series,)
        - "top_raw": the naive bottom-up sum (for comparison)
        - "top_independent": the independent top forecast that was fed in (for comparison)
    """
    n = len(bottom_forecasts)
    S = build_summing_matrix(n)  # (n+1, n)

    top_raw = float(np.sum(bottom_forecasts))
    top_input = float(top_forecast) if top_forecast is not None else top_raw
    y_hat = np.concatenate([[top_input], bottom_forecasts])  # (n+1,)

    if bottom_variances is None:
        bottom_variances = np.ones(n)
    # Structural scaling: top-level variance defaults to sum of bottom
    # variances (assuming independence across series) unless an explicit
    # top_variance is supplied for the independently-modeled top forecast.
    resolved_top_variance = top_variance if top_variance is not None else np.sum(bottom_variances)
    variances = np.concatenate([[resolved_top_variance], bottom_variances])
    W_inv = np.diag(1.0 / (variances + 1e-8))

    # y_reconciled = S (S' W^-1 S)^-1 S' W^-1 y_hat
    StWinv = S.T @ W_inv
    G = np.linalg.pinv(StWinv @ S) @ StWinv  # (n, n+1) projection matrix
    bottom_reconciled = G @ y_hat  # (n,)
    top_reconciled = float(np.sum(bottom_reconciled))  # guaranteed to equal the sum, by construction

    return {
        "top_raw": top_raw,
        "top_independent": top_input,
        "top_reconciled": top_reconciled,
        "bottom_reconciled": bottom_reconciled,
    }


def reconcile_forecast_matrix(bottom_forecast_matrix: np.ndarray, bottom_variances: np.ndarray = None,
                               top_forecast_series: np.ndarray = None, top_variance: float = None) -> dict:
    """
    Convenience wrapper: reconciles an ENTIRE forecast horizon at once.

    bottom_forecast_matrix: shape (n_series, output_window) -- e.g. each
        row is one SKU/store's P50 forecast across the next N days.
    bottom_variances: shape (n_series,) -- optional, applied to every
        horizon step (a series' relative uncertainty is assumed stable
        across the short forecast horizon, which is a reasonable
        simplification here).
    top_forecast_series: shape (output_window,) -- optional, an
        independently-modeled top-level forecast for each horizon step
        (e.g. from a separate SARIMA/LSTM fit directly on TOTAL demand
        rather than per-series). If omitted, falls back to the bottom-up
        sum at each step.

    Returns a dict with "bottom_reconciled" (n_series, output_window) and
    "top_reconciled" / "top_raw" (output_window,) arrays.
    """
    n_series, output_window = bottom_forecast_matrix.shape
    bottom_reconciled = np.zeros_like(bottom_forecast_matrix)
    top_reconciled = np.zeros(output_window)
    top_raw = np.zeros(output_window)

    for h in range(output_window):
        top_h = float(top_forecast_series[h]) if top_forecast_series is not None else None
        result = reconcile_forecasts(bottom_forecast_matrix[:, h], bottom_variances,
                                      top_forecast=top_h, top_variance=top_variance)
        bottom_reconciled[:, h] = result["bottom_reconciled"]
        top_reconciled[h] = result["top_reconciled"]
        top_raw[h] = result["top_raw"]

    return {
        "bottom_reconciled": bottom_reconciled,
        "top_reconciled": top_reconciled,
        "top_raw": top_raw,
    }


if __name__ == "__main__":
    # Minimal smoke test: 4 "stores" with INDEPENDENT bottom-level forecasts
    # that sum to 98, but a SEPARATELY modeled top-level (total) forecast
    # of 110 -- a realistic disagreement between two independently trained
    # models. Reconciliation should blend these toward consistency rather
    # than just trusting one source blindly.
    bottom_forecasts = np.array([20.0, 35.0, 15.0, 28.0])  # sums to 98
    bottom_variances = np.array([4.0, 9.0, 2.0, 6.0])  # different confidence per store
    independent_top_forecast = 110.0  # a separately-modeled total forecast that disagrees with the sum
    top_variance = 5.0  # this top model is fairly confident (low variance) -> should pull bottom up

    result = reconcile_forecasts(bottom_forecasts, bottom_variances,
                                  top_forecast=independent_top_forecast, top_variance=top_variance)
    print("Raw bottom forecasts:", bottom_forecasts, "-> naive sum:", result["top_raw"])
    print("Independent top-level forecast (disagrees with sum):", result["top_independent"])
    print("Reconciled bottom forecasts:", result["bottom_reconciled"])
    print("Reconciled top forecast:", result["top_reconciled"])
    print("Check: reconciled bottom sums to reconciled top?",
          np.isclose(result["bottom_reconciled"].sum(), result["top_reconciled"]))
    print("(Reconciled bottom values are pulled UP from the raw 98 total, toward the confident")
    print(" top-level forecast of 110 -- this is reconciliation actually doing something.)")

    # Matrix (multi-horizon) smoke test with an independent top-level series
    forecast_matrix = np.array([
        [20.0, 21.0, 19.0],
        [35.0, 34.0, 36.0],
        [15.0, 16.0, 14.0],
        [28.0, 27.0, 29.0],
    ])
    top_series = np.array([110.0, 105.0, 108.0])  # independent top-level model's forecast per horizon step
    matrix_result = reconcile_forecast_matrix(forecast_matrix, bottom_variances,
                                               top_forecast_series=top_series, top_variance=top_variance)
    print("\nMulti-horizon raw bottom-up sums:   ", matrix_result["top_raw"])
    print("Multi-horizon independent top series:", top_series)
    print("Multi-horizon reconciled top forecast:", matrix_result["top_reconciled"])
    print("Multi-horizon reconciled bottom forecasts:\n", matrix_result["bottom_reconciled"])
