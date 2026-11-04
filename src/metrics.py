
import numpy as np


def pinball_loss(y_true, y_pred, quantile):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    if y.shape != p.shape:
        raise ValueError("Shapes must match")

    error = y - p

    return float(
        np.mean(
            np.maximum(
                quantile * error,
                (quantile - 1) * error
            )
        )
    )


def scaled_pinball_loss(
    y_true,
    y_pred,
    quantile,
    insample,
    seasonality=7
):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    x = np.asarray(insample, dtype=float)

    if len(x) <= seasonality:
        raise ValueError("Insample series too short")

    scale = np.mean(
        np.abs(x[seasonality:] - x[:-seasonality])
    )

    return pinball_loss(y, p, quantile) / max(scale, 1e-8)


def interval_score(y_true, lower, upper, alpha=0.2):
    y = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    width = upper - lower

    score = (
        width
        + (2 / alpha) * (lower - y) * (y < lower)
        + (2 / alpha) * (y - upper) * (y > upper)
    )

    return float(np.mean(score))
