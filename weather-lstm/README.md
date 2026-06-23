# Weather Forecasting — LSTM

Direct multi-step recurrent forecaster for the 5-model group comparison. Produces
the shared **24 h × 7-variable** forecast on the real scale, rolled over the 2025
test year. The reported forecast is the **mean ensemble of three seeds**.

- **Input:** `(120, n_features)` window — 5-day lookback of the weather targets +
  cyclical time encodings.
- **Targets (7):** `temp, rh, dew, pressure, cloud, wx, wy` → output `(24, 7)`
  (single dense layer → 168 nodes, reshaped).
- **Split (chronological):** train 2021–2023, val 2024, test 2025. Test windows are
  stepped by 24 h (non-overlapping); train/val by 1 h.
- **Training:** tuned architecture (`lstm_units=128, dropout=0.1, batch=64,
  lr=1e-3`) with adaptive LR; three seeds `{42, 123, 2026}`.

## Background

The LSTM is a recurrent network that reads the 120 h window step by step,
maintaining a gated **cell state** that learns what to remember and what to forget
across the sequence — well suited to the long diurnal/synoptic dependencies in
weather. It is a **direct multi-step** forecaster: a dense head emits all
$24\times7=168$ values at once (no recursive feedback), which keeps long-horizon
error from compounding. The reported forecast is the mean ensemble of three seeds.

## Formula

At each step $t$ the cell processes input $x_t$ and previous state $h_{t-1}$
(concatenated). The gates and state updates are:

$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$$
$$\tilde{c}_t=\tanh(W_c[h_{t-1},x_t]+b_c),\quad o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$$
$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t,\qquad h_t=o_t\odot\tanh(c_t)$$

where $\sigma$ is the logistic sigmoid and $\odot$ the element-wise product. The
final hidden state $h_{120}$ is mapped to the forecast through two dense layers and
reshaped:

$$h_{\text{dense}}=\text{ReLU}(W_{d1}h_{120}+b_{d1}),\qquad \hat{y}_{\text{flat}}=W_{d2}h_{\text{dense}}+b_{d2}\ \rightarrow\ (24\times7).$$

## Architecture

```mermaid
flowchart LR
    X["Window (120×12)"] --> L["LSTM (128 units)<br/>return_sequences=False"]
    L --> D["Dropout 0.1"]
    D --> H["Dense (ReLU)"]
    H --> O["Dense 168 (linear)"]
    O --> R["Reshape → (24×7)"]
    R --> ENS["Mean over 3 seeds"]
```

## Dependencies

Python **3.12** (tested on 3.12.4). Runs on CPU (TensorFlow 2.21 on native Windows
is CPU-only); a CUDA GPU is used automatically if present.

```bash
pip install -r weather-lstm/requirements.txt
# or:
pip install "tensorflow>=2.15" "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.2" "matplotlib>=3.7" "statsmodels>=0.14"
```

## Data

Expects the shared hourly dataset at **`../data/weather_dataset.csv`** (repo-root
`data/` folder). Loading, wind decomposition, scaling and windowing are handled in
`src/data_prep.py`.

## Run (from the `weather-lstm/` folder)

```bash
cd weather-lstm
# UTF-8 needed on Windows consoles for the progress logs:
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python run.py
```

Trains the three seeds, builds the ensemble, evaluates, and runs diagnostics.
Training all three seeds takes roughly 40 min on CPU.

## Reproduce the reported results

After running, check `results_lstm/` (`per_variable_metrics.csv`,
`per_horizon_metrics.csv`). Expected on the 2025 test year (real scale, 3-seed
ensemble):

| Metric | Value |
|--------|-------|
| Overall MAE  | **≈5.39** (reruns land ≈5.34, within the bootstrap CI) |
| Overall RMSE | **≈12.01** |

Per-variable MAE: temp 1.11, rh 5.65, dew 1.06, pressure 1.33, cloud 21.42,
wx 4.18, wy 2.96.

## Outputs (`weather-lstm/results_lstm/`)

```
per_variable_metrics.csv   per-target MAE/RMSE/MAPE/R2 (+ bootstrap CIs)
per_horizon_metrics.csv    error vs lead time (h = 1..24)
diagnostics.txt            Ljung-Box / ADF / stability on h=1 residuals
y_test_real.npy            actuals  (N, 24, 7)
ensemble_pred.npy          ensemble forecasts (N, 24, 7)
                           — consumed by ../analysis/significance.py (DM/MCS tests)
plots/                     horizon error, residual normality, per-variable forecasts
```

## Layout

```
weather-lstm/
  run.py            orchestrator (run from this folder)
  src/
    config.py       paths, LOOKBACK=120, HORIZON=24, BEST_PARAMS, SEEDS=[42,123,2026]
    data_prep.py    load, wind decomposition, scale, sliding windows
    lstm_model.py   build/train, predict + inverse-transform
    evaluate.py     metrics, bootstrap CIs, per-horizon dump
    diagnostics.py  residual tests + result plots
```
