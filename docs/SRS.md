# Software Requirements Specification (SRS)
## Probabilistic Demand Forecasting with Uncertainty-Aware Inventory Decisions

**Version:** 1.0
**Prepared for:** Applied Scientist Internship Application Project
**Domain:** Time-Series Forecasting, Deep Learning, Statistical Inference, Decision Optimization

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a system that forecasts product-level demand **as a probability distribution** (not a single number) and uses that distribution to compute an optimal inventory decision. The system is designed to demonstrate applied competence in deep learning architecture, probability/statistics, and translating model output into a business decision.

### 1.2 Scope
The system will:
- Ingest historical daily/weekly sales data for multiple SKUs.
- Train a classical statistical baseline and one deep learning model.
- Produce calibrated prediction intervals (not just point forecasts).
- Statistically validate whether the DL model outperforms the baseline.
- Convert the forecast distribution into a safety-stock/reorder-point recommendation using the newsvendor model.

### 1.3 Intended Audience
Internship application reviewers, technical evaluators, and the author (as an implementation blueprint).

### 1.4 Definitions
| Term | Meaning |
|---|---|
| SKU | Stock Keeping Unit — a single product |
| Quantile Regression | Regression predicting a specific percentile of the target distribution instead of the mean |
| Pinball Loss | Loss function used to train quantile regression models |
| Calibration | Whether predicted intervals contain the true value at the stated frequency (e.g., 90% intervals cover 90% of true values) |
| Newsvendor Model | Classical inventory model that computes optimal stock level from a demand distribution and cost ratio |
| Diebold-Mariano Test | Statistical test to compare forecast accuracy between two models |

---

## 2. Overall Description

### 2.1 Product Perspective
A self-contained research/engineering project, not a production service. It runs as offline batch pipelines (data → training → evaluation → decision), reproducible via scripts/notebooks.

### 2.2 Product Functions (Summary)
1. Data ingestion and preprocessing
2. Baseline statistical forecasting (SARIMA)
3. Deep learning forecasting with quantile outputs (LSTM + pinball loss)
4. Uncertainty calibration evaluation
5. Statistical significance testing between models
6. Inventory decision layer (newsvendor optimization)
7. Reporting/visualization of results

### 2.3 Design Philosophy
Keep the architecture **simple and well-understood** rather than novel or complex. Accuracy and correctness of statistical claims matter more than architectural sophistication. One DL model, done rigorously, is sufficient.

---

## 3. Functional Requirements

### FR-1: Data Ingestion & Preprocessing
- FR-1.1: System shall load historical sales data (date, SKU id, quantity sold, price, optional promo/holiday flags).
- FR-1.2: System shall handle missing values and zero-demand days (common in retail — do not silently drop them, since zeros are informative).
- FR-1.3: System shall create sliding-window sequences (e.g., past 28 days → next 7 days) for supervised training.
- FR-1.4: System shall split data chronologically into train/validation/test (no random shuffling — time-series leakage must be avoided).

### FR-2: Baseline Model
- FR-2.1: System shall fit a SARIMA (or Exponential Smoothing) model per SKU or per SKU-group as a classical baseline.
- FR-2.2: System shall check stationarity (ADF test) and apply differencing if required before fitting.
- FR-2.3: Baseline shall output point forecasts and a Gaussian-based confidence interval for comparison purposes only.

### FR-3: Deep Learning Model
- FR-3.1: System shall implement a single LSTM (or GRU) sequence-to-sequence model. (No transformer required — keep it simple.)
- FR-3.2: Instead of predicting the mean, the model shall output **three quantiles** (e.g., 10th, 50th, 90th percentile) of demand for each forecast day.
- FR-3.3: System shall train using **pinball (quantile) loss**, summed across the three quantile heads.
- FR-3.4: Model input features: past demand window, day-of-week, price, and promo/holiday flags if available.

### FR-4: Uncertainty Calibration
- FR-4.1: System shall compute **empirical coverage**: the % of true values falling within the predicted 10th–90th percentile band, on the test set.
- FR-4.2: System shall report calibration error = |target coverage (80%) − empirical coverage|.
- FR-4.3: System shall plot predicted intervals vs. actual demand for a sample of SKUs.

### FR-5: Statistical Model Comparison
- FR-5.1: System shall compute **pinball loss** and **MAE/RMSE (on median forecast)** for both baseline and DL model.
- FR-5.2: System shall run a **Diebold-Mariano test** to determine if the DL model's forecast errors are statistically significantly lower than the baseline's.
- FR-5.3: System shall check forecast residuals for autocorrelation using the **Ljung-Box test** (a well-specified model should have residuals close to white noise).

### FR-6: Inventory Decision Layer
- FR-6.1: Given a holding cost `h` and stockout/shortage cost `s`, system shall compute the **critical ratio** `s / (s + h)`.
- FR-6.2: System shall use the predicted demand quantile corresponding to the critical ratio as the recommended **order-up-to level** (this is the newsvendor solution — directly usable since the model already outputs quantiles, no distribution assumption needed).
- FR-6.3: System shall simulate total cost (holding + stockout) under the DL model's recommendation vs. the baseline's recommendation, on held-out test data.

### FR-7: Reporting
- FR-7.1: System shall produce a summary report/notebook containing: model comparison table, calibration plot, DM test result, and cost simulation result.

---

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Reproducibility: fixed random seeds, versioned dependencies (requirements.txt) |
| NFR-2 | Runtime: full pipeline (data → decision) should run in under 1 hour on a single GPU or CPU for a moderate dataset (e.g., ~500 SKUs, 2 years daily data) |
| NFR-3 | Modularity: preprocessing, modeling, evaluation, and decision layers shall be separate, independently testable modules/scripts |
| NFR-4 | Documentation: each module shall have a docstring explaining statistical assumptions made (e.g., why quantile regression was chosen over Gaussian likelihood) |
| NFR-5 | Correctness over novelty: no component should sacrifice statistical validity for architectural complexity |

---

## 5. System Architecture (Simplified)

```
Raw Sales Data
      │
      ▼
[Preprocessing]  → chronological train/val/test split, windowing
      │
      ├──────────────► [SARIMA Baseline] ─────► point forecast + CI
      │
      ▼
[LSTM + Quantile Heads] ─────► P10, P50, P90 forecast per day
      │
      ▼
[Calibration Check] (coverage, PIT)
      │
      ▼
[Statistical Comparison] (DM test, Ljung-Box)
      │
      ▼
[Newsvendor Decision Layer] → recommended stock level
      │
      ▼
[Cost Simulation Report]
```

Notes on simplicity:
- One DL architecture (LSTM), not multiple competing architectures.
- Quantile regression is chosen over MC Dropout/Bayesian NN because it is simpler to implement correctly, faster to train, and directly usable in the newsvendor formula without extra distributional assumptions — this is a deliberate, defensible design choice, not a shortcut.

---

## 6. Data Requirements

- **Recommended dataset:** M5 Forecasting (Walmart) subset, or Rossmann Store Sales — both public, well-documented, and realistic (intermittent demand, promotions, holidays).
- **Minimum fields:** date, SKU/store id, units sold.
- **Optional fields (improves realism):** price, promotion flag, holiday flag.
- **Scale for this project:** 50–500 SKUs is sufficient; full dataset (30k+ SKUs) is not necessary and adds engineering overhead without adding scientific value.

---

## 7. Evaluation Metrics (Definition of Done)

| Metric | Purpose | Target |
|---|---|---|
| Pinball Loss | Quantile forecast accuracy | Lower than baseline |
| MAE / RMSE (P50) | Point forecast accuracy | Lower than baseline |
| Empirical Coverage (80% interval) | Calibration quality | Within ~75–85% |
| Diebold-Mariano p-value | Statistical significance of improvement | p < 0.05 to claim DL model is better |
| Ljung-Box p-value on residuals | Model adequacy | p > 0.05 (residuals ≈ white noise) |
| Total simulated cost | Business impact | Lower than baseline's cost |

---

## 8. Assumptions & Constraints

- Demand data may be intermittent (many zero-sales days); this is handled via quantile regression rather than assuming Gaussian errors.
- The project assumes access to at least 1–2 years of historical data per SKU to allow meaningful train/test splits.
- Costs `h` (holding) and `s` (stockout) for the newsvendor layer are assumed/set by the author (clearly stated as an assumption) since real cost data is usually proprietary.
- No real-time/streaming requirement — batch processing only.

---

## 9. Deliverables Checklist

- [ ] Cleaned, windowed dataset pipeline
- [ ] SARIMA baseline with stationarity checks
- [ ] LSTM quantile regression model (pinball loss)
- [ ] Calibration plot + coverage metric
- [ ] Diebold-Mariano + Ljung-Box test implementation
- [ ] Newsvendor decision function + cost simulation
- [ ] Final report/notebook summarizing all results
- [ ] README explaining statistical reasoning behind each design choice

---

## 10. Why This Satisfies "Accurate Results, Not Overcomplicated"

- Only **one DL model** is required (LSTM) — no need for transformers or ensembles.
- Quantile regression avoids incorrect distributional assumptions while staying simple to implement.
- Every statistical test used (ADF, Ljung-Box, Diebold-Mariano) is standard, well-documented, and has established Python implementations (`statsmodels`), so correctness is achievable without custom derivations.
- The newsvendor formula is a closed-form solution — no optimization solver needed, just reading off a predicted quantile.
