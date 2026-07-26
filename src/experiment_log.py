"""
experiment_log.py
------------------
A lightweight experiment tracker. Every training run (a full pipeline run,
a hyperparameter search trial, an ablation trial, or a CV fold) appends one
row to reports/experiment_log.csv instead of only printing to console and
disappearing.

This is intentionally simple -- a CSV, not a database or an MLflow server --
because at this project's scale a CSV is sufficient and transparent, and
adding real experiment-tracking infrastructure would be scope creep for
what's meant to demonstrate modeling/statistics skill, not MLOps tooling.
"""

import csv
from pathlib import Path
from datetime import datetime, timezone

LOG_PATH = Path("reports/experiment_log.csv")

FIELDNAMES = [
    "timestamp", "run_type", "dataset", "tag",
    "hidden_size", "input_window", "output_window", "n_features", "feature_cols",
    "epochs", "fold",
    "pinball_loss", "mae", "empirical_coverage", "calibration_error",
    "dm_p_value", "ljung_box_p_value",
    "notes",
]


def log_experiment(record: dict, log_path: Path = LOG_PATH):
    """
    Append one experiment result to the log. Missing fields are written as
    empty strings so the CSV stays a consistent shape across different run
    types (a hyperparameter trial won't have a dm_p_value, an ablation
    trial won't have a fold number, etc.).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    row = {field: record.get(field, "") for field in FIELDNAMES}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_experiment_log(log_path: Path = LOG_PATH):
    """Load the experiment log as a pandas DataFrame (imported lazily to keep this module light)."""
    import pandas as pd
    if not log_path.exists():
        return pd.DataFrame(columns=FIELDNAMES)
    return pd.read_csv(log_path)


if __name__ == "__main__":
    # Minimal smoke test
    log_experiment({
        "run_type": "smoke_test", "dataset": "synthetic", "tag": "test",
        "hidden_size": 64, "input_window": 28, "output_window": 7,
        "n_features": 1, "feature_cols": "[]", "epochs": 5,
        "pinball_loss": 1.23, "mae": 2.34, "empirical_coverage": 0.81,
        "calibration_error": 0.01, "notes": "smoke test row",
    })
    df = load_experiment_log()
    print(df.tail())
