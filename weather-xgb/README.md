# Weather Forecasting — XGBoost

XGBoost member of the group comparison defined in [`../PROTOCOL.md`](../PROTOCOL.md).
Same data handling, split, horizon and metrics as the VAR / DL models so results
are directly comparable.

## What it does

Produces the shared artifact: a **24 h × 7-variable** forecast on the real scale,
evaluated by rolling 24 h origins over the 2025 test year.

- **Input** `(120, 11)` — 7 weather features (the 7 targets) + 4 cyclical time
  encodings. **Deviation from §2:** the protocol specifies 8 weather features
  (12 total); by project decision `precip` is dropped, giving 11. Agree the
  shared input set across the group for a fair comparison.
- **Targets (7)** `temp, rh, dew, pressure, cloud, wx, wy` → output `(24, 7)`.
- **Model** — XGBoost is tabular, so the `(120, 11)` window is encoded into
  compact engineered features (recent lags + rolling mean/std/min/max per
  channel) and **7 multi-output regressors** (one per target) each predict that
  variable's 24 horizons. Every feature is a function of the 120 h history up to
  the origin only — same information and no-leakage horizon as the raw window.
- **Split** chronological: train 2021–2023, val 2024, test 2025. Never shuffled.
- **Scaling** `StandardScaler` fit on **train only**; forecasts denormalised before
  scoring. Cyclical time encodings are left in `[-1, 1]`.
- **Early stopping** on 2024 val windows; the 2025 test set is untouched during
  training.
- **Determinism** seeds `[0, 1, 2]`; metrics reported for the seed-ensemble plus a
  per-seed std.

## Layout (mirrors the VAR reference, §10)

```
weather-xgb/
  run.py                  # orchestrator
  src/
    config.py             # paths, features, split, window & XGB params, seeds
    data_prep.py          # load, clean, wind decomp, time encodings, split, scale, windows
    xgb_model.py          # 7 multi-output regressors, early stopping, save
    diagnostics.py        # §6 ML/DL analogues (convergence, Ljung-Box, normality, intervals)
    evaluate.py           # §7 rolling-origin metrics, bootstrap CIs, plots
  results/
    metrics.csv           # overall / per-horizon / per-variable MAE & RMSE (+ CIs)
    diagnostics.txt        # §6 diagnostics report
    plots/                # learning curves, residuals, per-horizon, pred-vs-actual
  models/                 # saved boosters + training history per seed
```

## Run

```bash
python weather-xgb/run.py            # full: 3 seeds, dense train windows
python weather-xgb/run.py --smoke    # fast validation (1 seed, small model)
```

## Diagnostics

| Protocol method      | XGBoost analogue implemented |
|----------------------|------------------------------|
| Stability            | val-RMSE learning curves converge, no divergence |
| Ljung-Box (lag 24)   | autocorrelation of 1-step val-forecast residuals, per variable |
| Residual normality   | D'Agostino K² test + histogram / Q-Q plots |
| Forecast intervals   | empirical residual bands that widen with the horizon (+ coverage) |
| Bootstrap CI         | resample test origins → 95% CI on overall MAE/RMSE |

## Metrics

`results/metrics.csv` holds **overall**, **per-horizon (1…24)** and **per-variable**
MAE & RMSE. Overall MAE/RMSE additionally carry a bootstrap 95% CI and a multi-seed std.
