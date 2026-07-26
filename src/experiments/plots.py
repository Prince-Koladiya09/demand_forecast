"""
experiments/plots.py
-----------------------
Shared plotting functions for the four experiment scripts. Centralized here
so hyperparam_search.py, ablation_study.py, rolling_cv.py, and
segment_analysis.py all produce consistent, comparable chart styling instead
of each hand-rolling its own.

Every function saves a PNG to reports/figures/ AND returns the matplotlib
figure, so it works both when run as a script (headless, just saves) and
inside a notebook (displays inline).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

FIGURES_DIR = Path("reports/figures")
sns.set_style("whitegrid")


def _save(fig, name, save_dir=None):
    save_dir = Path(save_dir) if save_dir is not None else FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")
    return path


def plot_hyperparam_heatmap(summary_df, metric="pinball_loss", save_dir=None):
    """Heatmap of pinball loss across hidden_size x input_window grid."""
    pivot = summary_df.pivot(index="hidden_size", columns="input_window", values=metric)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", ax=ax, cbar_kws={"label": metric})
    ax.set_title(f"Hyperparameter Search: {metric} by hidden_size x input_window\n(lower is better)")
    plt.tight_layout()
    _save(fig, "hyperparam_search_heatmap", save_dir=save_dir)
    return fig


def plot_ablation_comparison(summary_df, save_dir=None):
    """Side-by-side bars comparing target-only vs. with-extra-features on pinball loss and coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(data=summary_df, x="config", y="pinball_loss", hue="config", legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title("Pinball Loss (lower is better)")
    axes[0].set_xlabel("")

    sns.barplot(data=summary_df, x="config", y="empirical_coverage", hue="config", legend=False, ax=axes[1], palette="viridis")
    axes[1].axhline(0.80, color="red", linestyle="--", linewidth=1, label="Target (80%)")
    axes[1].set_title("Empirical Coverage (closer to 80% target is better)")
    axes[1].set_xlabel("")
    axes[1].legend()

    fig.suptitle("Feature Ablation: Target-Only vs. With Extra Features")
    plt.tight_layout()
    _save(fig, "ablation_comparison", save_dir=save_dir)
    return fig


def plot_rolling_cv_folds(pinball_scores, coverage_scores, save_dir=None):
    """Line plot of pinball loss and coverage across CV folds, showing stability (or lack of it)."""
    folds = list(range(1, len(pinball_scores) + 1))
    mean_pinball = np.mean(pinball_scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(folds, pinball_scores, marker="o", color="tab:blue")
    axes[0].axhline(mean_pinball, color="gray", linestyle="--", linewidth=1, label=f"Mean ({mean_pinball:.2f})")
    axes[0].set_title("Pinball Loss per Fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_xticks(folds)
    axes[0].legend()

    axes[1].plot(folds, coverage_scores, marker="o", color="tab:orange")
    axes[1].axhline(0.80, color="red", linestyle="--", linewidth=1, label="Target (80%)")
    axes[1].set_title("Empirical Coverage per Fold")
    axes[1].set_xlabel("Fold")
    axes[1].set_xticks(folds)
    axes[1].legend()

    fig.suptitle("Rolling-Origin Cross-Validation: Stability Across Time Periods")
    plt.tight_layout()
    _save(fig, "rolling_cv_folds", save_dir=save_dir)
    return fig


def plot_segment_results(segment_df, save_dir=None):
    """Bar chart comparing pinball loss and coverage across low/medium/high volume SKU buckets."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    order = [b for b in ["low", "medium", "high"] if b in segment_df["volume_bucket"].values]

    sns.barplot(data=segment_df, x="volume_bucket", y="pinball_loss", hue="volume_bucket", legend=False, order=order, ax=axes[0], palette="viridis")
    axes[0].set_title("Pinball Loss by SKU Volume Segment")
    axes[0].set_xlabel("Volume Segment")

    sns.barplot(data=segment_df, x="volume_bucket", y="empirical_coverage", hue="volume_bucket", legend=False, order=order, ax=axes[1], palette="viridis")
    axes[1].axhline(0.80, color="red", linestyle="--", linewidth=1, label="Target (80%)")
    axes[1].set_title("Empirical Coverage by SKU Volume Segment")
    axes[1].set_xlabel("Volume Segment")
    axes[1].legend()

    fig.suptitle("Per-SKU Segment Analysis")
    plt.tight_layout()
    _save(fig, "segment_analysis_comparison", save_dir=save_dir)
    return fig


def plot_experiment_log_overview(log_df, metric="pinball_loss", top_n=15, save_dir=None):
    """
    Bar chart ranking ALL logged runs (across every run_type) by a chosen
    metric -- useful as a single at-a-glance view after running several
    experiment scripts, not just one at a time.
    """
    df = log_df.dropna(subset=[metric]).copy()
    df["label"] = df["run_type"] + " / " + df["tag"].astype(str)
    df = df.sort_values(metric).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df))))
    sns.barplot(data=df, y="label", x=metric, hue="run_type", dodge=False, ax=ax, palette="tab10")
    ax.set_title(f"Top {top_n} Logged Runs by {metric} (lower is better)")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "experiment_log_overview", save_dir=save_dir)
    return fig
