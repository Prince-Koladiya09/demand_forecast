"""
calibration.py
--------------
Implements FR-4: Uncertainty Calibration.

Checks whether the predicted intervals are actually trustworthy --
i.e., does the 80% predicted interval (P10-P90) really contain the
true value roughly 80% of the time? This is the check that most
forecasting projects skip, and the one that actually validates the
model's uncertainty estimates (not just its point accuracy).
"""

import numpy as np
import matplotlib.pyplot as plt


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Fraction of true values that fall within [lower, upper].
    For an 80% interval (P10-P90), a well-calibrated model should have
    empirical coverage close to 0.80.
    """
    within_interval = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within_interval))


def calibration_error(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, target_coverage=0.80) -> float:
    """Absolute difference between target and empirical coverage."""
    return abs(empirical_coverage(y_true, lower, upper) - target_coverage)


def plot_forecast_with_interval(dates, y_true, median, lower, upper, title="Forecast vs Actual", save_path=None):
    """
    Plot true demand against the predicted median and P10-P90 band.
    Save to reports/figures/ if save_path is given.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual demand", color="black", linewidth=1.5)
    plt.plot(dates, median, label="Predicted median (P50)", color="tab:blue")
    plt.fill_between(dates, lower, upper, color="tab:blue", alpha=0.2, label="P10-P90 interval")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    rng = np.random.default_rng(0)
    y_true = rng.poisson(20, size=100).astype(float)
    median = y_true + rng.normal(0, 2, size=100)
    lower = median - 5
    upper = median + 5

    cov = empirical_coverage(y_true, lower, upper)
    err = calibration_error(y_true, lower, upper)
    print(f"Empirical coverage: {cov:.3f}")
    print(f"Calibration error (vs 80% target): {err:.3f}")
