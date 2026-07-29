# Probabilistic Demand Forecasting with Uncertainty-Aware Inventory Decisions

An applied-science project demonstrating deep learning, probability/statistics, and
decision-making under uncertainty — built for demand forecasting.

Instead of predicting a single number ("50 units next week"), this project predicts a
**distribution** over demand (via quantile regression) and uses that distribution to make
a real inventory decision (via the newsvendor model), while statistically validating every
claim of improvement along the way.

Full requirements and design rationale: [`docs/SRS.md`](docs/SRS.md)

---

## Project Structure

```
demand_forecasting_project/
│
├── docs/
│   └── SRS.md                     # Full software requirements specification
│
├── data/
│   ├── raw/                       # Place raw dataset here (e.g., M5 or Rossmann CSVs)
│   └── processed/                 # Cached cleaned/windowed data (see "Processed Data" below)
│
├── src/
│   ├── data_loaders/
│   │   ├── rossmann_loader.py     # Converts raw Rossmann CSV -> standard schema
│   │   └── m5_loader.py           # Melts M5 wide format -> standard schema
│   ├── experiments/
│   │   ├── common.py              # Shared train_and_evaluate() used by all experiment scripts below
│   │   ├── plots.py               # Shared plotting functions for all experiment scripts
│   │   ├── hyperparam_search.py   # Compares hidden_size x input_window configs
│   │   ├── ablation_study.py      # Compares target-only vs. target+extra-features
│   │   ├── rolling_cv.py          # Rolling-origin (walk-forward) cross-validation
│   │   ├── segment_analysis.py    # Breaks down results by per-SKU demand volume tier
│   │   └── advanced_analysis.py   # Ties together uncertainty decomposition, structural
│   │                               # break detection, and hierarchical reconciliation (see below)
│   ├── experiment_log.py          # Appends every run's metrics to reports/experiment_log.csv
│   ├── preprocessing.py           # FR-1: load, clean, window, chronological split, cache to data/processed/
│   ├── baseline_sarima.py         # FR-2: SARIMA baseline + ADF stationarity check
│   ├── dl_model.py                # FR-3: LSTM quantile regression model (pinball loss, dropout for MC Dropout)
│   ├── calibration.py             # FR-4: coverage, PIT, calibration plots
│   ├── statistical_tests.py       # FR-5: Diebold-Mariano test, Ljung-Box test
│   ├── decision_newsvendor.py     # FR-6: critical ratio, order-up-to level, cost simulation
│   ├── uncertainty_decomposition.py    # Epistemic vs. aleatoric uncertainty via MC Dropout
│   ├── structural_break_detection.py   # CUSUM-based regime shift detection + pruning
│   ├── hierarchical_reconciliation.py  # Structural-scaling (MinT-style) forecast reconciliation
│   └── pipeline.py                # Orchestrates the full pipeline end-to-end
│
├── notebooks/
│   ├── 00_eda.ipynb                       # Exploratory data analysis (see "EDA" below)
│   ├── 01_end_to_end_walkthrough.ipynb    # Interactive run of the full pipeline
│   └── 02_experiment_results.ipynb        # Visualizes results from reports/experiment_log.csv
│
├── reports/
│   └── figures/                   # Calibration plots, forecast plots saved here
│
├── tests/
│   └── test_smoke.py              # Basic smoke tests to check each module imports/runs
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Set up environment
```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add data
Loaders are provided for both **Rossmann Store Sales** and **M5 Forecasting (Walmart)** —
they convert each dataset's raw format into the standard schema
(`date, sku_id, units_sold`, plus extra feature columns) that the rest of the
pipeline expects, and now also merge in each dataset's extra metadata/price files
as real model inputs (not just unused columns).

**Rossmann — place 2 files in `data/raw/`:**
- `train.csv` — core daily sales
- `store.csv` — store metadata (StoreType, Assortment, CompetitionDistance, Promo2)

Both from [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales).
Handled by `src/data_loaders/rossmann_loader.py`, which drops closed-store days
(`Open == 0`), merges `store.csv` on `Store`, label-encodes `StoreType`/`Assortment`,
and fills missing `CompetitionDistance` with the max observed distance (treating
"unknown" as "far away" rather than 0). Produces 5 extra feature channels:
`promo, competition_distance, promo2, store_type_code, assortment_code`.

**M5 — place 3 files in `data/raw/`:**
- `sales_train_validation.csv` — core daily sales (wide format)
- `calendar.csv` — maps day IDs to real dates and weekly price IDs
- `sell_prices.csv` — weekly price per item-store

All three from [Kaggle - M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
Handled by `src/data_loaders/m5_loader.py`, which melts M5's wide (one-column-per-day)
format into long format, merges weekly `sell_price` via `wm_yr_wk`, and subsamples
series (default: 300) to keep training fast — full history length per series is
preserved, only the number of series is reduced. Produces 1 extra feature channel:
`sell_price`.

Files **not** used (and why): `sample_submission.csv` (Kaggle submission template,
not data), `test.csv` (Rossmann's held-out competition test set, has no ground truth
to evaluate against), `sales_train_evaluation.csv` (optional extended M5 test window,
skipped to keep the project simple).

### 3. Run the full pipeline
```bash
python -m src.pipeline --dataset rossmann   # or: --dataset m5
```
The LSTM's input size adjusts automatically to however many feature channels
the chosen dataset's loader provides (1 target channel + N extra features).

This will:
1. Preprocess data and create chronological train/val/test splits
2. Fit the SARIMA baseline (with ADF stationarity check)
3. Train the LSTM quantile regression model (P10/P50/P90)
4. Evaluate calibration (coverage) for both models
5. Run the Diebold-Mariano test (DL vs. baseline) and Ljung-Box test (residual diagnostics)
6. Compute the newsvendor-optimal stock level and simulate total cost
7. Save all plots to `reports/figures/` and print a summary table to the console

### 4. Explore interactively
Open `notebooks/01_end_to_end_walkthrough.ipynb` to run each step interactively and
inspect intermediate results/plots.

---

## Processed Data (`data/processed/`)

Preprocessing is the slow part of this project (especially M5's wide-to-long melt
and price merge) — training and exploring the data is what you actually do
repeatedly. So `src/pipeline.py` automatically **caches** the output of
preprocessing here every time it runs, instead of recomputing it every time:

```
data/processed/
├── {dataset}_train.npz              # X, y, sku arrays for the training split
├── {dataset}_val.npz                # same, for validation
├── {dataset}_test.npz                # same, for test
└── {dataset}_cleaned_long.parquet   # cleaned long-format DataFrame (pre-windowing)
                                       # -- this is what the EDA notebook loads
```

To reload cached windows later without rerunning preprocessing:
```python
from src.preprocessing import load_processed_windows, load_processed_long_df

X_train, y_train, sku_train = load_processed_windows("rossmann", split="train")
df = load_processed_long_df("rossmann")   # cleaned long-format, used for EDA/plotting
```

---

## Exploratory Data Analysis (`notebooks/00_eda.ipynb`)

Run this **after** `python -m src.pipeline --dataset <name>` at least once, since
it reads from `data/processed/{dataset}_cleaned_long.parquet`.

It covers, with saved plots in `reports/figures/`:

1. **Overview & missingness** — sanity-checks that `fill_missing()` actually left no gaps
2. **Demand distribution** — histogram + skewness, showing *why* this project uses
   quantile regression instead of assuming a Gaussian/MSE-style error
3. **Time series trends** — raw demand plotted per sample SKU
4. **Day-of-week seasonality** — boxplot of demand by weekday, motivating SARIMA's
   `seasonal_order=(...,7)` choice in `src/baseline_sarima.py`
5. **ACF / PACF plots** — statistically confirms weekly seasonality (a spike at
   lag 7), rather than just eyeballing it
6. **Promo / price effect** — checks whether the extra feature channels (`promo`
   for Rossmann, `sell_price` for M5) actually correlate with demand
7. **Intermittency check** — % of zero-demand days per SKU, another justification
   for quantile regression over a Gaussian-assumption model

Change `DATASET_NAME = 'rossmann'` to `'m5'` in the first code cell to explore
the other dataset (assuming you've run the pipeline for it at least once too).

---

## Experiments

Beyond the main pipeline (one model, one train/val/test split), four experiment
scripts answer specific questions a single run can't. All of them log their
results to **`reports/experiment_log.csv`** via `src/experiment_log.py`, alongside
every run of the main pipeline — so every run type (main pipeline, hyperparameter
trial, ablation config, CV fold, segment analysis) ends up in one comparable table
instead of scattered console output that disappears.

### Hyperparameter comparison
```bash
python -m src.experiments.hyperparam_search --dataset rossmann
```
Trains across a small grid of `hidden_size x input_window` combinations and
ranks them by pinball loss (not MAE/RMSE — pinball loss evaluates the whole
predicted distribution, which is the right metric for a quantile model; see
FR-5.1 in `docs/SRS.md`). Deliberately a small grid (3x2), not an exhaustive
search — the point is showing a search space was considered and compared
fairly, not running an expensive tuning job. Saves a heatmap to
`reports/figures/hyperparam_search_heatmap.png`.

### Feature ablation study
```bash
python -m src.experiments.ablation_study --dataset rossmann
```
Trains the identical model twice — once with only the target channel, once
with the full extra-feature set (`sell_price` for M5; `promo`,
`competition_distance`, etc. for Rossmann) — and reports whether those
features actually improved pinball loss. Answers "did the features I added
actually help," honestly, rather than assuming they did because they were
added. Saves a comparison chart to `reports/figures/ablation_comparison.png`.

### Rolling-origin cross-validation
```bash
python -m src.experiments.rolling_cv --dataset rossmann --n_folds 3
```
A single train/val/test split only tells you how the model did on ONE test
period, which could have been unusually easy or hard by chance. This expands
the training window forward across several folds (walk-forward validation —
the time-series-safe alternative to random k-fold CV, which would leak future
data into training) and reports the mean **and standard deviation** of pinball
loss across folds. A high std means performance is unstable across time
periods — worth knowing before trusting a single split's result. Saves a
per-fold chart to `reports/figures/rolling_cv_folds.png`.

### Per-SKU segmentation
```bash
python -m src.experiments.segment_analysis --dataset rossmann
```
A single blended pinball loss across all SKUs can hide very different
behavior for different types of series. This buckets SKUs into low/medium/high
volume tertiles (by total historical demand) and reports pinball loss, MAE,
and coverage **per bucket** — typically revealing that the model performs
worse on low-volume, intermittent SKUs than on high-volume ones, which the
blended average alone would hide. Saves a comparison chart to
`reports/figures/segment_analysis_comparison.png`.

### Advanced analysis (uncertainty, structural breaks, reconciliation)
```bash
python -m src.experiments.advanced_analysis --dataset rossmann
```
Trains one model, then runs three deeper analyses on top of it:

**1. Uncertainty decomposition** (`src/uncertainty_decomposition.py`) — splits
total predictive uncertainty into **epistemic** (model uncertainty, reducible
with more data) and **aleatoric** (inherent demand noise, not reducible) via
Monte Carlo Dropout: dropout is normally OFF at inference, MC Dropout keeps
it ON and runs ~20 stochastic forward passes per input, using the spread
across passes as an epistemic-uncertainty estimate, and the model's own
predicted quantile spread (P90−P10) as an aleatoric-uncertainty proxy. This
is a well-established but *approximate* technique (Gal & Ghahramani, 2016),
not a full Bayesian treatment — stated explicitly in the module docstring.
Reveals whether forecast uncertainty is mostly "the model hasn't seen enough
examples" vs. "this demand is just inherently noisy," which call for
different responses. Saves `reports/figures/uncertainty_decomposition.png`.

**2. Structural break detection** (`src/structural_break_detection.py`) —
runs two-sided CUSUM (cumulative sum) control-chart detection per SKU to
find sustained regime shifts (e.g. a competitor opening, a permanent promo
change). CUSUM parameters were empirically tuned (see module docstring) to
give zero false positives on pure-noise series while detecting an injected
shift with near-zero delay. If a break is found, retrains the model with
pre-break data pruned and compares pinball loss against the unpruned model,
reporting whether the older regime was adding noise or useful signal. Saves
`reports/figures/regime_pruning_comparison.png` (only if at least one break
was detected).

**3. Hierarchical reconciliation** (`src/hierarchical_reconciliation.py`) —
takes each SKU's latest forecast plus an **independently fit** SARIMA model
on the aggregated total demand (a genuinely separate model, not just a sum),
then reconciles the two so they agree exactly. Implements the standard
hierarchical reconciliation formula directly in numpy using a "structural
scaling" (diagonal-weight) simplification of full MinT reconciliation —
*note: there is no `statsmodels.tsa.forecasting.reconciliation` module, so
this is implemented from the formula directly rather than depending on an
API that doesn't exist*. Saves `reports/figures/reconciliation_comparison.png`.

### Reviewing results
```python
from src.experiment_log import load_experiment_log
df = load_experiment_log()
df.sort_values("pinball_loss")
```
Or open `notebooks/02_experiment_results.ipynb` for a visual view of all of
the above — it reloads every chart saved by the scripts above, plus one
additional cross-experiment chart ranking all logged runs together.

---

## Design Notes (Why It's Kept Simple)

- **One baseline (SARIMA), one DL model (LSTM).** No ensembles, no transformers — added
  complexity would not add scientific value here.
- **Quantile regression is the primary uncertainty method, not a full Bayesian model.**
  Simpler to implement correctly, faster to train, and its output (quantiles) plugs
  directly into the newsvendor formula with no extra distributional assumptions. MC
  Dropout is used *in addition*, specifically in `src/uncertainty_decomposition.py`, to
  answer a different question (epistemic vs. aleatoric uncertainty) that quantile
  regression alone can't answer — it's an addition for a specific diagnostic purpose,
  not a replacement for the main approach.
- **Every statistical test used (ADF, Diebold-Mariano, Ljung-Box) is a standard, well-implemented
  test in `statsmodels`/`scipy`.** No custom statistics were derived — the value is in choosing
  the *correct* test for each specific claim, not in reinventing one.
- **Hierarchical reconciliation uses a simplified "structural scaling" weight matrix,
  not full MinT covariance estimation.** Full MinT needs substantial in-sample residual
  history to estimate a full covariance matrix reliably; the diagonal approximation used
  here is less statistically optimal but far more stable at this project's data scale,
  while still guaranteeing the actual business requirement (forecasts sum consistently).

See `docs/SRS.md`, Section 10, for the full reasoning behind each design decision.

---

## Key Files to Read First

If you're reviewing this project (e.g., as an interviewer), read in this order:
1. `docs/SRS.md` — full spec and reasoning
2. `src/dl_model.py` — the core model and pinball loss implementation
3. `src/statistical_tests.py` — DM test and Ljung-Box, with inline comments on *why* each is used
4. `src/decision_newsvendor.py` — how the forecast becomes a business decision
5. `src/uncertainty_decomposition.py`, `src/structural_break_detection.py`,
   `src/hierarchical_reconciliation.py` — the three advanced analyses, each with an
   explicit statement of what method/approximation is used and why