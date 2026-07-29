"""
structural_break_detection.py
--------------------------------
Detects structural breaks (regime shifts) in a demand series -- e.g. a new
competitor opening, a permanent promo-strategy change, or COVID-era demand
shifts (relevant to M5, which spans 2011-2016 and would span the COVID
period in the extended/evaluation version of the dataset).

Method: two-sided CUSUM (cumulative sum) control chart, a standard,
well-understood statistical process control technique for detecting a
sustained shift in a series' mean -- not a novel method, but the right
level of complexity for this project (see docs/SRS.md's "accurate results,
not overcomplicated" design philosophy). A more elaborate alternative
(Bayesian online change-point detection) exists but adds complexity without
adding much value at this project's scale.

How CUSUM works, briefly:
    Standardize the series, then accumulate positive and negative
    deviations from a target (drift) separately:
        cusum_pos[t] = max(0, cusum_pos[t-1] + z[t] - k)
        cusum_neg[t] = min(0, cusum_neg[t-1] + z[t] + k)
    where z[t] is the standardized value and k is a small "drift"
    allowance (how much deviation to tolerate before it counts as
    evidence of a shift). If either cumulative sum exceeds a threshold h,
    a break is flagged at that point -- a sustained shift, not a single
    noisy day, is what triggers detection.

Once a break is found, the recommended action is regime pruning: discard
training data from BEFORE the break, since it reflects a demand pattern
that no longer applies -- keeping it would just add noise/bias to training,
not signal.
"""

import numpy as np
import pandas as pd


def cusum_break_detection(series: np.ndarray, k: float = 1.0, threshold: float = 8.0, min_segment: int = 40):
    """
    Two-sided CUSUM structural break detection.

    series: 1D array of a single SKU's demand history, in time order.
    k: drift allowance in standard deviations (how much deviation before
       it starts accumulating as evidence of a shift). Default 1.0 was
       chosen empirically (see module tests) to give zero false positives
       on pure-noise series while still detecting a true shift with near-
       zero delay -- a lower k (e.g. 0.5) is more sensitive but triggers
       false breaks from ordinary noise runs.
    threshold: how many cumulative standard deviations of sustained
       deviation trigger a detected break. Default 8.0, tuned alongside k
       above. Larger threshold = fewer, more confident breaks.
    min_segment: minimum number of days a series needs before a break can
       be considered valid -- avoids flagging spurious "breaks" before the
       series has a stable baseline, and ensures the baseline mean/std
       estimate isn't itself too noisy.

    Returns: break_index (int) if a break is found, else None.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < min_segment * 2:
        return None  # not enough history to reliably detect + still have a usable post-break segment

    mean = series[:min_segment].mean()
    std = series[:min_segment].std() + 1e-8
    z = (series - mean) / std

    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))

    for t in range(1, len(series)):
        cusum_pos[t] = max(0, cusum_pos[t - 1] + z[t] - k)
        cusum_neg[t] = min(0, cusum_neg[t - 1] + z[t] + k)

        if t >= min_segment and (cusum_pos[t] > threshold or cusum_neg[t] < -threshold):
            # Require enough data remains AFTER the break to be usable
            if len(series) - t >= min_segment:
                return t

    return None


def detect_breaks_per_sku(df: pd.DataFrame, sku_col: str, date_col: str, target_col: str,
                           k: float = 1.0, threshold: float = 8.0, min_segment: int = 40):
    """
    Runs cusum_break_detection() per SKU in a long-format DataFrame.
    Returns a dict: {sku_id: break_date or None}
    """
    breaks = {}
    for sku, group in df.groupby(sku_col):
        group = group.sort_values(date_col)
        series = group[target_col].values
        break_idx = cusum_break_detection(series, k=k, threshold=threshold, min_segment=min_segment)
        breaks[sku] = group[date_col].values[break_idx] if break_idx is not None else None
    return breaks


def apply_regime_pruning(df: pd.DataFrame, sku_col: str, date_col: str, breaks: dict) -> pd.DataFrame:
    """
    Given a dict of {sku_id: break_date_or_None} from detect_breaks_per_sku(),
    returns a copy of df with each SKU's pre-break rows removed. SKUs with
    no detected break are left untouched.
    """
    pruned_frames = []
    for sku, group in df.groupby(sku_col):
        break_date = breaks.get(sku)
        if break_date is not None:
            group = group[group[date_col] >= break_date]
        pruned_frames.append(group)
    return pd.concat(pruned_frames, ignore_index=True)


if __name__ == "__main__":
    # Minimal smoke test: a series with an obvious mean shift halfway through
    rng = np.random.default_rng(0)
    before = rng.normal(20, 3, 100)
    after = rng.normal(50, 3, 100)  # clear upward shift
    series = np.concatenate([before, after])

    break_idx = cusum_break_detection(series)
    print(f"Detected break at index: {break_idx} (true break is at index 100)")

    # Per-SKU DataFrame smoke test
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "sku_id": ["A"] * 200 + ["B"] * 200,
        "units_sold": np.concatenate([series, rng.normal(20, 3, 200)]),  # B has no break
    })

    breaks = detect_breaks_per_sku(df, sku_col="sku_id", date_col="date", target_col="units_sold")
    print("Detected breaks per SKU:", breaks)

    pruned = apply_regime_pruning(df, sku_col="sku_id", date_col="date", breaks=breaks)
    print(f"Original rows: {len(df)}, pruned rows: {len(pruned)}")
