
import numpy as np


def cqr_adjust(lower, upper, y_cal, alpha=0.2):
    """
    Conformalized Quantile Regression.

    IMPORTANT:
    y_cal must be from a dedicated calibration split.
    Do not use training, validation or final test data here.
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y_cal, dtype=float)

    if not (lower.shape == upper.shape == y.shape):
        raise ValueError("Input shapes must match")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    scores = np.maximum(
        lower - y,
        y - upper
    )

    n = len(scores)

    rank = min(
        n,
        int(np.ceil((n + 1) * (1 - alpha)))
    )

    adjustment = float(
        np.partition(scores, rank - 1)[rank - 1]
    )

    return {
        "adjustment": adjustment,
        "lower": lower - adjustment,
        "upper": upper + adjustment,
        "nominal_coverage": 1 - alpha,
        "calibration_size": n
    }
