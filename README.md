
# Probabilistic Demand Forecasting with Uncertainty-Aware Inventory Decisions

### Version 2.0 | Deep Learning | Statistical Forecasting | Inventory Optimization

An applied machine learning and operations research project that combines probabilistic demand forecasting, statistical evaluation, uncertainty quantification, and inventory optimization.

The objective is not simply to predict future demand. It is to understand demand uncertainty and convert forecasts into better inventory decisions.

Instead of predicting a single value (e.g., 50 units next week), the system predicts multiple demand quantiles (P10, P50, P90) using quantile regression and uses these predictions to calculate inventory decisions through the Newsvendor model.

Version 2 extends the original implementation with improved preprocessing, monotonic quantile predictions, statistical evaluation utilities, conformal calibration, and multi-period inventory simulation.

Full requirements and design rationale: `docs/SRS.md`

---

# 1. Project Objectives

The project addresses five main objectives:

1. Build reliable demand forecasting models using historical sales data.
2. Predict demand distributions rather than only point forecasts.
3. Measure predictive uncertainty and interval calibration.
4. Statistically evaluate forecasting performance against baseline models.
5. Convert probabilistic forecasts into inventory ordering decisions and evaluate their financial impact.

## Main Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Quantile LSTM |
| NumPy | Numerical computation |
| Pandas | Data processing |
| SciPy | Statistical testing |
| Statsmodels | SARIMA, ETS and diagnostics |
| Scikit-learn | Preprocessing and evaluation |
| Matplotlib / Seaborn | Visualizations |
| Pytest | Automated testing |
| GitHub Actions | Continuous integration |

---

# 2. Datasets

The project supports two demand forecasting datasets.

## 2.1 Rossmann Store Sales

Dataset: [Kaggle – Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data)

Required files:

- `train.csv`
- `store.csv`

Important features:

- Historical store sales
- Promotions
- Store type
- Assortment
- Competition distance
- Calendar information
- Store operating status

The Rossmann loader merges store metadata with historical sales and prepares the data in a common forecasting schema.

## 2.2 M5 Forecasting Accuracy

Dataset: [Kaggle – M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Required files:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Important features:

- Historical item sales
- Store identifiers
- Item identifiers
- Selling prices
- Calendar information
- Weekly price information

The M5 loader converts wide-format sales data into long-format time-series data and merges calendar and selling-price information.

## 2.3 Dataset Directory

Place downloaded datasets in:

```text
data/
├── raw/
│   ├── train.csv
│   ├── store.csv
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
│
└── processed/
```

Use the files relevant to the selected dataset.

Do not commit large raw datasets to GitHub unless explicitly required.

---

# 3. Updated Project Structure

```text
demand_forecasting_project/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── SRS.md
│
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_end_to_end_walkthrough.ipynb
│   └── 02_experiment_results.ipynb
│
├── reports/
│   ├── figures/
│   ├── results.md
│   └── experiment_log.csv
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   ├── rossmann_loader.py
│   │   └── m5_loader.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── plots.py
│   │   ├── hyperparam_search.py
│   │   ├── ablation_study.py
│   │   ├── rolling_cv.py
│   │   ├── segment_analysis.py
│   │   └── advanced_analysis.py
│   │
│   ├── preprocessing.py
│   ├── baseline_sarima.py
│   ├── dl_model.py
│   ├── calibration.py
│   ├── statistical_tests.py
│   ├── decision_newsvendor.py
│   ├── experiment_log.py
│   ├── pipeline.py
│   │
│   ├── metrics.py
│   ├── advanced_evaluation.py
│   ├── conformal.py
│   ├── inventory_simulation.py
│   │
│   ├── uncertainty_decomposition.py
│   ├── structural_break_detection.py
│   └── hierarchical_reconciliation.py
│
├── tests/
│   ├── test_smoke.py
│   └── test_v2.py
│
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
```

The structure retains the original project modules and adds the v2 functionality. Existing notebooks, documentation and figures should be preserved.

---

# 4. Version 2 Improvements

## 4.1 Data Preprocessing

Updated functionality:

- Improved Rossmann and M5 data loading.
- Consistent time-series schema.
- Chronological data splitting.
- Training-only feature scaling.
- Improved missing-value handling.
- Dataset preprocessing utilities.

Expected value:

- Reduces data leakage.
- Improves input consistency.
- Makes model training more reproducible.

## 4.2 Quantile LSTM Improvements

File: `src/dl_model.py`

The LSTM predicts multiple demand quantiles.

Version 2 improvements:

- Monotonic quantile outputs.
- Quantile crossing prevention.
- Validation-based early stopping.
- Best checkpoint restoration.
- Global random seed support.
- Configurable model parameters.

Default quantiles:

```python
QUANTILES = [0.1, 0.5, 0.9]
```

The model produces:

- P10: Lower demand estimate.
- P50: Median demand estimate.
- P90: Upper demand estimate.

Quantile predictions represent different levels of demand uncertainty.

Note: Existing v1 checkpoints should be retrained when using the updated model architecture.

## 4.3 Forecast Evaluation Metrics

File: `src/metrics.py`

Implemented metrics:

- Pinball Loss
- Scaled Pinball Loss
- Interval Score

These metrics evaluate both forecast accuracy and the quality of predicted demand intervals.

## 4.4 Forecasting Baselines

Files:

- `src/advanced_evaluation.py`
- `src/baseline_sarima.py`

Forecasting methods include:

- Seasonal Naive
- Exponential Smoothing (ETS)
- SARIMA
- Quantile LSTM
- Quantile lag-feature baseline utilities

Baseline comparisons help determine whether the deep learning model provides meaningful improvements over simpler forecasting methods.

## 4.5 Statistical Evaluation

File: `src/statistical_tests.py`

Statistical methods include:

- Diebold-Mariano test
- Horizon-aware HAC variance estimation
- Harvey-Leybourne-Newbold correction
- Ljung-Box residual diagnostic
- ADF stationarity testing through the SARIMA module

These methods support statistical evaluation of forecasting errors.

For fair comparisons, models should use identical forecast origins, target observations and evaluation periods.

## 4.6 Newsvendor Inventory Optimization

File: `src/decision_newsvendor.py`

The Newsvendor model converts probabilistic forecasts into inventory decisions.

Critical ratio:

```python
critical_ratio = stockout_cost / (
    holding_cost + stockout_cost
)
```

Version 2 improvements:

- Quantile interpolation.
- Per-SKU ordering recommendations.
- Holding cost calculation.
- Stockout cost calculation.
- Fill-rate evaluation.
- Order-level recommendations.

Example:

```python
from src.decision_newsvendor import recommend_order_level

predictions = {
    0.1: 10,
    0.5: 20,
    0.9: 30
}

result = recommend_order_level(
    predictions,
    holding_cost=1.0,
    stockout_cost=4.0
)

print(result)
```

Expected recommended order level:

```text
27.5
```

The critical ratio is 0.80, so the recommendation is interpolated between P50 and P90.

## 4.7 Inventory Simulation

File: `src/inventory_simulation.py`

Version 2 introduces multi-period inventory simulation.

Features:

- Lead-time support.
- Inventory replenishment.
- Holding costs.
- Stockout costs.
- Fill-rate calculation.
- Average inventory calculation.
- Period-level simulation records.

Example:

```python
from src.inventory_simulation import simulate_order_up_to

result = simulate_order_up_to(
    demand=[5, 10, 8],
    forecasts=[8, 12, 10],
    lead_time=0,
    holding_cost=1.0,
    stockout_cost=4.0
)

print(result["total_cost"])
print(result["fill_rate"])
```

This provides a framework for evaluating inventory decisions over multiple periods.

## 4.8 Conformal Prediction Calibration

File: `src/conformal.py`

Version 2 includes split conformalized quantile regression.

Features:

- Calibration residual scores.
- Quantile interval adjustment.
- Configurable coverage level.
- Calibration sample size reporting.

Example:

```python
from src.conformal import cqr_adjust

result = cqr_adjust(
    lower=[10, 12, 14],
    upper=[20, 22, 24],
    y_cal=[21, 25, 19],
    alpha=0.2
)

print(result["adjustment"])
print(result["lower"])
print(result["upper"])
```

Important: Calibration observations must be kept separate from training and final test observations.

---

# 5. Existing Advanced Features

The original project includes three additional research components.

## 5.1 Uncertainty Decomposition

File: `src/uncertainty_decomposition.py`

Uses Monte Carlo Dropout to estimate:

- Epistemic uncertainty: Model uncertainty.
- Aleatoric uncertainty: Inherent demand variability.

The implementation uses repeated stochastic forward passes and quantile spread as an approximation of uncertainty.

## 5.2 Structural Break Detection

File: `src/structural_break_detection.py`

Uses CUSUM-based change detection to identify potential demand regime shifts.

Main functions:

- `cusum_break_detection`
- `detect_breaks_per_sku`
- `apply_regime_pruning`

This helps investigate whether historical demand patterns remain representative of current demand.

## 5.3 Hierarchical Forecast Reconciliation

File: `src/hierarchical_reconciliation.py`

Supports reconciliation between aggregate forecasts and individual series forecasts.

Main functions:

- `reconcile_forecasts`
- `reconcile_forecast_matrix`

The current implementation uses structural-scaling reconciliation rather than full MinT covariance estimation.

---

# 6. Installation and Environment Setup

## 6.1 Requirements

Recommended:

- Python 3.11 or 3.12
- Git
- VS Code
- Internet connection for dependency installation

## 6.2 Clone the Repository

```bash
git clone YOUR_EXISTING_GITHUB_REPOSITORY_URL

cd demand_forecasting_project
```

## 6.3 Create a Virtual Environment

Windows:

```powershell
python -m venv venv

venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv

source venv/bin/activate
```

## 6.4 Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

## 6.5 Verify Python Imports

```bash
python -c "import src; print(src.__file__)"
```

If successful, Python can locate the project package.

---

# 7. Running the Project

## 7.1 Run the Main Forecasting Pipeline

Rossmann:

```bash
python -m src.pipeline --dataset rossmann
```

M5:

```bash
python -m src.pipeline --dataset m5
```

The original pipeline is designed to:

1. Load and preprocess data.
2. Generate chronological train, validation and test splits.
3. Train the forecasting model.
4. Evaluate forecasting performance.
5. Calculate prediction coverage.
6. Run statistical diagnostics.
7. Generate inventory recommendations.
8. Save figures and experiment results.

The newly added v2 utilities require integration into the main pipeline before they are automatically executed by these commands.

## 7.2 Run Hyperparameter Search

```bash
python -m src.experiments.hyperparam_search --dataset rossmann
```

Compares different model configurations.

## 7.3 Run Feature Ablation

```bash
python -m src.experiments.ablation_study --dataset rossmann
```

Evaluates the contribution of additional input features.

## 7.4 Run Rolling-Origin Cross-Validation

```bash
python -m src.experiments.rolling_cv --dataset rossmann --n_folds 3
```

Evaluates forecasting performance across multiple chronological folds.

## 7.5 Run Demand Segmentation

```bash
python -m src.experiments.segment_analysis --dataset rossmann
```

Evaluates performance across low-, medium- and high-volume demand segments.

## 7.6 Run Advanced Analysis

```bash
python -m src.experiments.advanced_analysis --dataset rossmann
```

Runs the existing advanced analysis workflow for uncertainty decomposition, structural breaks and hierarchical reconciliation.

---

# 8. Automated Testing

The project includes original smoke tests and additional v2 regression tests.

Test files:

```text
tests/
├── test_smoke.py
└── test_v2.py
```

Run all tests:

```bash
python -m pytest -v
```

Run original tests:

```bash
python -m pytest tests/test_smoke.py -v
```

Run v2 tests:

```bash
python -m pytest tests/test_v2.py -v
```

The tests cover:

- Quantile LSTM output shape.
- Quantile monotonicity.
- Pinball loss.
- Empirical coverage.
- Diebold-Mariano testing.
- Ljung-Box diagnostics.
- Newsvendor critical ratio.
- Quantile interpolation.
- Inventory simulation.
- Conformal calibration.
- Rolling-origin splitting.
- Demand segmentation.
- Monte Carlo Dropout.
- Structural break detection.
- Hierarchical reconciliation.

Tests validate software behavior; passing tests alone do not establish scientific superiority or real-world inventory savings.

---

# 9. Experiment Configuration

Configuration file:

`configs/default.yaml`

Example:

```yaml
project: probabilistic-demand-forecasting-v2

seed: 42
dataset: rossmann

input_window: 28
forecast_horizon: 7

quantiles:
  - 0.05
  - 0.10
  - 0.25
  - 0.50
  - 0.75
  - 0.90
  - 0.95

training:
  hidden_size: 64
  epochs: 50
  patience: 7
  learning_rate: 0.001

evaluation:
  seeds: [11, 22, 33, 44, 55]
  horizons: [1, 7, 14, 28]
  origins: 4

inventory:
  holding_cost: 1.0
  stockout_cost: 4.0
  lead_time_days: 1
```

The configuration defines the intended experiment settings.

The model and experiment runners must explicitly consume these values for the configuration to control execution.

---

# 10. Processed Data and Caching

The preprocessing pipeline uses `data/processed/` for generated datasets and cached features.

Expected structure:

```text
data/processed/
├── rossmann_train.npz
├── rossmann_val.npz
├── rossmann_test.npz
└── rossmann_cleaned_long.parquet
```

The exact generated files depend on the dataset and preprocessing implementation.

Processed files should not be committed unless needed for reproducibility or explicitly requested.

---

# 11. Experiment Results and Reporting

Results are stored under:

```text
reports/
├── figures/
├── results.md
└── experiment_log.csv
```

Important evaluation measures:

| Metric | Purpose |
|---|---|
| Pinball Loss | Quantile forecast accuracy |
| MAE | Absolute forecast error |
| RMSSE | Scaled forecasting accuracy |
| Interval Coverage | Prediction interval reliability |
| Interval Score | Coverage and interval width |
| DM Test | Statistical comparison of forecasts |
| Total Inventory Cost | Financial inventory performance |
| Fill Rate | Demand satisfaction |
| Average Inventory | Inventory holding level |

All model comparisons should use common evaluation periods and identical demand observations.

No performance improvement, statistical significance or inventory savings should be claimed without actual experiment results.

---

# 12. Git and GitHub Development Workflow

The v2 changes are organized into 12 feature-wise commits.

| Commit | Feature |
|---|---|
| 1 | Rossmann and M5 preprocessing improvements |
| 2 | Leakage-free feature scaling |
| 3 | Quantile LSTM improvements |
| 4 | Probabilistic evaluation metrics |
| 5 | Forecasting baselines |
| 6 | Statistical testing improvements |
| 7 | Newsvendor inventory optimization |
| 8 | Multi-period inventory simulation |
| 9 | Conformal prediction calibration |
| 10 | Smoke and regression tests |
| 11 | Experiment configuration and CI |
| 12 | README and reporting updates |

Create a feature branch:

```bash
git checkout -b feature/probabilistic-forecasting-v2
```

Check changes:

```bash
git status
```

Create feature-wise commits:

```bash
git add src/
git commit -m "Implement forecasting improvements"
```

Push the feature branch:

```bash
git push -u origin feature/probabilistic-forecasting-v2
```

Create a Pull Request into the existing `main` branch.

The original Git history is preserved.

---

# 13. Continuous Integration

GitHub Actions configuration:

`.github/workflows/tests.yml`

The intended workflow automatically:

1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies.
4. Executes pytest.
5. Reports test status.

This helps detect regressions when new code is pushed or submitted through a pull request.

---

# 14. Known Limitations and Future Work

The current v2 implementation adds core functionality, but the following research and engineering work remains:

- Full fixed-origin orchestration across every model.
- Complete LightGBM quantile benchmark.
- Store and SKU learned embeddings.
- Explicit closed-day loss masking.
- Intermittent-demand hurdle or negative-binomial model.
- Deep ensemble training.
- Fully integrated five-seed and multi-horizon experiments.
- Full hierarchical reconciliation experiments.
- Structural-break false-positive analysis across real datasets.
- End-to-end conformal calibration integration.
- Complete inventory policy comparisons.
- Interactive forecasting dashboard.
- Inference API.
- Production drift monitoring.

These are future implementation and validation milestones, not completed features.

---

# 15. Research and Business Value

The project connects machine learning with practical inventory management.

Its main contribution is the use of probabilistic forecasting for decision-making under uncertainty.

Expected research value:

- Evaluation of quantile-based forecasting.
- Comparison against statistical forecasting baselines.
- Investigation of predictive uncertainty.
- Statistical validation of forecast comparisons.
- Calibration of prediction intervals.

Expected business value:

- Demand-aware inventory recommendations.
- Explicit consideration of holding and stockout costs.
- Lead-time-aware inventory simulation.
- Measurement of service levels and inventory costs.
- Better-informed ordering decisions.

Actual financial improvements must be established through controlled experiments on held-out demand data.

---

# 16. Key Files to Read First

For someone reviewing or evaluating the project:

1. `docs/SRS.md` – Software requirements and design rationale.
2. `src/dl_model.py` – Quantile LSTM implementation.
3. `src/preprocessing.py` – Data preparation and feature processing.
4. `src/metrics.py` – Probabilistic forecasting metrics.
5. `src/statistical_tests.py` – Statistical model evaluation.
6. `src/decision_newsvendor.py` – Inventory decision optimization.
7. `src/inventory_simulation.py` – Inventory policy simulation.
8. `src/conformal.py` – Prediction interval calibration.
9. `src/uncertainty_decomposition.py` – Uncertainty analysis.
10. `src/structural_break_detection.py` – Demand regime changes.
11. `src/hierarchical_reconciliation.py` – Forecast reconciliation.
12. `tests/` – Automated regression and functionality tests.

---

# 17. Final Project Goal

The goal of this project is to move beyond point forecasting.

It combines:

**Demand Forecasting + Predictive Uncertainty + Statistical Validation + Inventory Optimization**

The final objective is to demonstrate, through reproducible experiments, whether probabilistic demand forecasts can improve inventory decisions compared with conventional forecasting and ordering approaches.