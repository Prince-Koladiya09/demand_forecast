
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def diebold_mariano_test(
    errors_model_1,
    errors_model_2,
    h=1,
    power=2,
    loss_differential=None
):
    e1 = np.asarray(errors_model_1, dtype=float)
    e2 = np.asarray(errors_model_2, dtype=float)

    if e1.shape != e2.shape or e1.ndim != 1:
        raise ValueError("Inputs must be paired 1D arrays")

    if not np.all(np.isfinite(e1)) or not np.all(np.isfinite(e2)):
        raise ValueError("Errors must be finite")

    if loss_differential is None:
        d = np.abs(e1)**power - np.abs(e2)**power
    else:
        d = np.asarray(loss_differential, dtype=float)

    if d.shape != e1.shape or not np.all(np.isfinite(d)):
        raise ValueError("Invalid loss differential")

    n = len(d)

    if h < 1 or n <= h:
        raise ValueError("Require n > horizon >= 1")

    mean_d = float(d.mean())
    centered = d - mean_d

    variance = float(np.dot(centered, centered) / n)

    for lag in range(1, h):
        gamma = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
        variance += 2 * (1 - lag / h) * gamma

    if variance <= 0:
        return {
            "dm_statistic": float("nan"),
            "p_value": 1.0,
            "model_1_better": False,
            "model_2_better": False,
            "mean_loss_difference": mean_d,
            "n": n,
            "horizon": h
        }

    dm = mean_d / np.sqrt(variance / n)

    correction = np.sqrt(
        max(
            (n + 1 - 2*h + h*(h-1)/n) / n,
            0
        )
    )

    corrected_dm = dm * correction

    p_value = float(
        2 * stats.t.sf(abs(corrected_dm), df=n-1)
    )

    return {
        "dm_statistic": float(corrected_dm),
        "p_value": p_value,
        "model_1_better": mean_d < 0 and p_value < 0.05,
        "model_2_better": mean_d > 0 and p_value < 0.05,
        "mean_loss_difference": mean_d,
        "n": n,
        "horizon": h
    }


def ljung_box_test(residuals, lags=10):

    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim != 1:
        raise ValueError("Residuals must be one-dimensional")

    if len(residuals) <= lags:
        raise ValueError("Residual length must exceed lags")

    result = acorr_ljungbox(
        residuals,
        lags=[lags],
        return_df=True
    )

    statistic = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])

    return {
        "lb_statistic": statistic,
        "p_value": p_value,
        "residuals_are_white_noise": p_value > 0.05
    }
