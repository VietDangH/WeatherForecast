# Weather Forecasting — Shared Experiment Protocol

This document defines the **running protocol and evaluation metrics** for the group's
weather-forecasting comparison. Every model (VAR, classical ML, deep learning) MUST follow
the same data handling, split, horizon, and metrics so results are directly comparable.
Model-specific deviations are called out explicitly under each section.

---

## 1. Dataset

- Source: hourly weather export (Open-Meteo style), file `weather_dataset.csv`.
- Coverage: **2021-01-01 → 2025-12-31**, hourly, **43,824 rows, no gaps, no nulls**.
- Raw columns (clean name in parentheses):
  `temperature_2m (temp)`, `relative_humidity_2m (rh)`, `dew_point_2m (dew)`,
  `precipitation (precip)`, `surface_pressure (pressure)`, `cloud_cover (cloud)`,
  `cloud_cover_low/mid/high`, `wind_speed_10m (wind_speed)`, `wind_direction_10m (wind_dir)`,
  `wind_gusts_10m (wind_gust)`, `soil_moisture_0_to_7cm (soil)`.

> If your raw data is 10-minute resolution, resample to hourly (mean per hour) first.
> This dataset is already hourly, so no resampling is applied.

---

## 2. Preprocessing (identical for all models)

1. Parse `time`, set as index, sort ascending, enforce hourly frequency.
2. Sensor-error guard: clip wind columns to `>= 0` (handles any `-9999` artifacts).
3. **Wind decomposition** — convert circular wind to Cartesian so models can learn it:
   ```python
   wx = wind_speed * cos(radians(wind_dir))
   wy = wind_speed * sin(radians(wind_dir))
   ```
4. Any incidental gaps: time-interpolate, then forward/back fill.

### Target variables (7) — must match across the group
```
temp, rh, dew, pressure, cloud, wx, wy
```
Output shape per forecast: **(24, 7)** = 24 hours ahead × 7 variables.

### Input features
| Model family | Input features | Lookback |
|---|---|---|
| **VAR** | the **7 targets only** (no time encodings — not valid AR lags) | last `p` lags, `p` from AIC |
| **ML / DL** | 8 weather features + **4 cyclical time encodings** = **12** | fixed **120 h** window → input `(120, 12)` |

Cyclical time encodings (ML/DL input-only helpers):
`sin/cos` of hour-of-day and `sin/cos` of day-of-year.

---

## 3. Data split (chronological — NEVER shuffle)

| Split | Years | Hours | Share | Role |
|---|---|---|---|---|
| Train | 2021, 2022, 2023 | 26,280 | 60% | fit model parameters |
| Val   | 2024 | 8,784 | 20% | hyperparameter tuning / early stopping |
| Test  | 2025 | 8,760 | 20% | untouched until final evaluation |

- Same boundaries for **every** model so the comparison is fair.
- **VAR note:** fits on 2021–2023 and does not tune on 2024 (lag order via AIC on train).
  Decide once as a group whether the final VAR refits on 2021–2024 (use all pre-test data)
  or stays on 2021–2023 (strict train-boundary parity). State the choice in the report.
- **No leakage:** forecasts only ever condition on observations strictly before the target window.

---

## 4. Normalization

- Fit the scaler (e.g. `StandardScaler`) on **train only**; apply to val/test.
- **Denormalize forecasts before scoring** — all metrics are computed on real-scale values.
- Use the **same scaler strategy** across the group.
- VAR may run unscaled (scale-invariant); scale only if variances differ wildly and cause
  numerical issues. ML/DL should always scale.

---

## 5. Running protocol per model family

### 5.1 VAR (statistical)
1. ADF stationarity test per series on train. If non-stationary, difference (`d`, usually 1)
   and re-test; record `d`. *(On this dataset ADF reports stationarity → `d=0`, fit on levels.)*
2. AIC lag-order selection (`maxlags=48`) → `p`.
3. Fit `VAR(p)` once on train (OLS, deterministic — no seed).
4. Diagnostics before trusting forecasts: stability (`is_stable`), Ljung-Box on residuals
   (white noise at lag 24), residual normality. Increase `maxlags`/`d` if they fail.
5. Forecast recursively; invert differencing (if `d>0`) and the scaler before scoring.

### 5.2 ML / DL (e.g. XGBoost, LSTM, GRU, Transformer)
1. Build sliding windows: input `(120, 12)` → target `(24, 7)`.
2. Train on train windows; tune/early-stop on val (2024) windows.
3. Set and **report the random seed**; run multiple seeds where feasible (see §8).
4. Save the best checkpoint by validation metric; never touch test during training.

> Whatever the internal mechanics, every model produces the same artifact: a **24 h × 7-variable
> forecast** for a given origin, on the real scale, scored by the metrics in §6.

---

## 6. Post-estimation statistical methods (diagnostics)

Run **after fitting, before trusting forecasts**. These validate that the model is
well-specified. Each model reports a short diagnostics paragraph. `p > 0.05` = "good"
for the white-noise / normality tests.

| Method | Purpose | VAR | ML / DL analogue |
|---|---|---|---|
| **ADF test** (`adfuller`) | stationarity / unit root (pre-fit) | per series on train; sets `d` | n/a (handled by learning) |
| **Stability** (`is_stable`) | roots inside unit circle → forecasts won't diverge | required | check loss convergence / no divergence |
| **Ljung-Box** (`acorr_ljungbox`, lag 24) | residual autocorrelation — leftover structure means lag/capacity too low | per residual series | residual autocorrelation on val forecasts |
| **Residual normality** (`test_normality`) | validity of analytical intervals | joint test | residual histogram / Q-Q plot |
| **Forecast intervals** | uncertainty bands that widen over the horizon | analytical (`forecast_interval`) | quantile loss / MC-dropout / ensemble spread |
| **Bootstrap CI on metrics** | report MAE/RMSE ± uncertainty, not a bare point | resample forecast windows | same — applies to every model |

**Decision rule:** if Ljung-Box fails or the model is unstable, fix it before evaluating —
VAR: increase `maxlags` or difference further; ML/DL: revisit window length, features, or
capacity. Bootstrap CIs and forecast intervals apply to **all** model families so the
reported uncertainty is comparable.

---

## 7. Evaluation — rolling-origin over the test year

Evaluate all models with **rolling 24 h forecasts** across 2025 so coverage and protocol match:

- For each origin `t` stepping through 2025, condition on true history up to `t`,
  forecast `t+1 … t+24`, invert to real scale, compare to actuals.
- **Origin step:** `24 h` non-overlapping (default) — use the *same* step for every model.
  (Switch all models to `1 h` overlapping together if you want denser coverage.)
- VAR: fit once, roll the origin on ground-truth history (no per-origin refit).
- DL/ML: slide the input window the same way.

### Metrics (real / denormalized scale)
Let `ŷ` = forecast, `y` = actual, over horizons `h=1..24`, variables `v=1..7`, and `N` origins.

- **MAE (primary):**  `mean(|ŷ − y|)`
- **RMSE:**  `sqrt(mean((ŷ − y)²))`

Report all three breakdowns:
1. **Overall** — single MAE & RMSE averaged over all horizons, variables, origins.
2. **Per-horizon** (`h = 1 … 24`) — exposes error accumulation over the horizon.
3. **Per-variable** — some variables (e.g. `cloud`) forecast far worse than others.

Optional but encouraged: bootstrap 95% CIs on MAE/RMSE; forecast-interval bands (VAR provides
these analytically).

---

## 8. Determinism & reporting

- **VAR is deterministic** (OLS) — a single run is correct; state this, no multi-seed.
- **ML/DL are stochastic** — fix and report the seed; ideally report mean ± std over ≥3 seeds.
- Each model reports, on the **2025 test set**:
  - overall MAE & RMSE,
  - per-horizon error curve (plot),
  - per-variable error table,
  - predicted-vs-actual plots,
  - any diagnostics relevant to the model.

---

## 9. Invariants (do not violate)

- Test set (2025) is untouched until final rolling evaluation.
- Splits are chronological; never shuffle across time.
- Same target list (7), same horizon (24), same rolling step, same metrics for every model.
- Scaler fit on train only; denormalize before computing metrics.
- VAR uses the 7 weather targets only (no time encodings); ML/DL use the 12-feature input.
- Report results on the real scale, not normalized units.

---

## 10. Reference implementation

The VAR reference pipeline lives in [`weather-var/`](weather-var/):
`run.py` → `src/{config,data_prep,var_model,diagnostics,evaluate}.py`,
writing `results/metrics.csv`, `results/diagnostics.txt`, and `results/plots/`.
Run with: `python weather-var/run.py`.

---

## 11. EDA evidence for the design choices

The exploratory analysis backing the decisions above lives in two notebooks:
[`eda.ipynb`](eda.ipynb) (overview: distributions, weekly/seasonal plots, correlations,
wind rose) and [`eda_extra.ipynb`](eda_extra.ipynb) (diagnostics aligned to the VAR
assumptions). Key findings that justify the protocol:

| Finding (from `eda_extra.ipynb`) | Supports |
|---|---|
| **ADF + KPSS agree** the targets are level-stationary; rolling mean/variance are flat aside from an **annual cycle** (not a unit-root trend). | `d = 0` (fit on levels, §5.1) |
| **PACF** cuts off after a few lags with a clear **lag-24 spike** (daily cycle). | VAR lag order `p` (AIC + the 24 h structure, §5.1) |
| **Seasonal decomposition** isolates a strong daily cycle and a smooth annual component; residual is near-stationary. | optional seasonal exog (hour + day-of-year sin/cos) as an experiment |
| **Lagged cross-correlations** are strong off-diagonal (e.g. `temp`↔`dew`/`rh`, wind components). | multivariate VAR over 7 separate univariate models |

> `pressure` retains residual autocorrelation even at `maxlags=48` (long-memory) — expect it
> to be the weakest VAR target and flag it in the report rather than over-fitting the lag order.
