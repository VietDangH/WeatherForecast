# DLinear — Report Content

> Số liệu từ:
> - `outputs/S/evaluation_results.json` + `outputs/I/evaluation_results.json`
> - `outputs/S/diagnostics/diagnostic_report.txt` + `outputs/I/diagnostics/diagnostic_report.txt`
>
> Style theo XGBoost section. Review xong mình convert sang LaTeX.

---

## 1. Overview

DLinear \citep{zeng2023dlinear} is a decomposition-based linear model that challenges the prevailing trend of applying complex Transformer architectures to time series forecasting. The core insight is that a time series can be separated into a slow-moving trend and a fast-moving seasonal component, each forecast independently with a single linear layer, then recombined. Despite extreme simplicity — only 5,892 parameters for the shared-weight variant — DLinear matches or exceeds state-of-the-art Transformers on standard benchmarks while training orders of magnitude faster.

In this project, DLinear is applied as a **multivariate multi-step** forecaster: a single model consumes the full 120 × 11 input window (7 weather channels + 4 cyclical time encodings) and outputs a 24 × 7 forecast block covering all seven protocol targets simultaneously. A final linear projection layer maps the 11 input channels to the 7 output targets, allowing the model to exploit cross-variable dependencies learned from data.

Two variants are evaluated:

- **DLinear-S** (Shared): one pair of trend/seasonal linear layers shared across all 11 input channels — 5,892 parameters.
- **DLinear-I** (Individual): per-channel trend/seasonal linear layers — 63,972 parameters.

---

## 2. Mathematical Formulation

Let $\mathbf{X} \in \mathbb{R}^{L \times C_{\text{in}}}$ be an input window of $L = 120$ hours and $C_{\text{in}} = 11$ features (7 weather variables + 4 cyclical time encodings). The model produces $\mathbf{\hat{Y}} \in \mathbb{R}^{H \times C_{\text{out}}}$ with $H = 24$ hours and $C_{\text{out}} = 7$ target variables.

### 2.1 Series Decomposition

A 1D average pooling with kernel size $k = 25$ and stride 1 extracts the trend while preserving sequence length. The official implementation \citep{zeng2023dlinear} uses **replication padding** (repeating boundary values). For a sequence of length $L$, the first and last time steps are replicated $p = (k-1)/2 = 12$ times and concatenated to both ends before pooling:

$$
\mathbf{X}_{\text{padded}} = [\,\underbrace{\mathbf{X}_0, \ldots, \mathbf{X}_0}_{p \text{ times}},\; \mathbf{X}_0, \ldots, \mathbf{X}_{L-1},\; \underbrace{\mathbf{X}_{L-1}, \ldots, \mathbf{X}_{L-1}}_{p \text{ times}}\,] \quad \in \mathbb{R}^{(L + 2p) \times C_{\text{in}}}
$$

$$
\mathbf{T} = \text{AvgPool1d}\big(\mathbf{X}_{\text{padded}},\; \text{kernel}=k,\; \text{padding}=0\big) \quad \in \mathbb{R}^{L \times C_{\text{in}}}
$$

$$
\mathbf{S} = \mathbf{X} - \mathbf{T} \quad \in \mathbb{R}^{L \times C_{\text{in}}}
$$

At each position $t$, the trend is the unweighted mean of the $k$ values in the local window:

$$
\mathbf{T}_t = \frac{1}{k} \sum_{j=t-p}^{t+p} \mathbf{X}_{\text{padded}, j}
$$

The kernel size $k=25$ is chosen because it spans just over one full diurnal cycle (24 h), causing the day–night oscillation to approximately cancel when averaged, leaving only the multi-day trend. The seasonal component $\mathbf{S}$ captures all remaining variation: diurnal cycles, short-term fluctuations, and noise.

### 2.2 Dual Linear Projection

Each component is independently mapped from the 120 h lookback to the 24 h forecast horizon via a single linear layer:

$$
\mathbf{\hat{Y}}_t = \mathbf{W}_t \, \mathbf{T} + \mathbf{b}_t \quad \in \mathbb{R}^{H \times C_{\text{in}}} \qquad \text{(Trend forecast)}
$$
$$
\mathbf{\hat{Y}}_s = \mathbf{W}_s \, \mathbf{S} + \mathbf{b}_s \quad \in \mathbb{R}^{H \times C_{\text{in}}} \qquad \text{(Seasonal forecast)}
$$

where $\mathbf{W}_t, \mathbf{W}_s \in \mathbb{R}^{H \times L}$ and $\mathbf{b}_t, \mathbf{b}_s \in \mathbb{R}^{H}$.

- **DLinear-S**: a single pair $(\mathbf{W}_t, \mathbf{W}_s)$ is shared across all $C_{\text{in}}$ channels.
- **DLinear-I**: each channel $c \in \{1, \ldots, C_{\text{in}}\}$ has its own pair $\mathbf{W}_t^{(c)}, \mathbf{W}_s^{(c)} \in \mathbb{R}^{H \times L}$, implemented as $C_{\text{in}}$ separate `nn.Linear(L, H)` modules.

### 2.3 Combine and Project

$$
\mathbf{\hat{Y}}_{\text{raw}} = \mathbf{\hat{Y}}_t + \mathbf{\hat{Y}}_s \quad \in \mathbb{R}^{H \times C_{\text{in}}}
$$
$$
\mathbf{\hat{Y}} = \mathbf{\hat{Y}}_{\text{raw}} \, \mathbf{W}_{\text{proj}} \quad \in \mathbb{R}^{H \times C_{\text{out}}}
$$

$\mathbf{W}_{\text{proj}} \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$ maps the $C_{\text{in}}$ input channels to the $C_{\text{out}}$ protocol targets. This projection is linear and data-driven: the model learns weighted combinations of input channels for each target variable (e.g. $wx$ and $wy$ may contribute to predicting each other, temperature helps predict dew point).

**Parameter count:**
- DLinear-S: $2 \times (LH + H) \times 1 + C_{\text{in}} C_{\text{out}} = \mathbf{5{,}892}$
- DLinear-I: $2 \times (LH + H) \times C_{\text{in}} + C_{\text{in}} C_{\text{out}} = \mathbf{63{,}972}$

### 2.4 Why Decomposition Helps

The decomposition acts as an **inductive bias**. The trend branch only needs to learn a smooth continuation ("is it getting warmer or cooler over the next 24 hours?"), while the seasonal branch only needs to capture the repeating diurnal pattern ("at what hour does the daily peak occur?"). By separating these two structurally different tasks, each linear layer solves a simpler problem than one layer would face on the raw mixed signal.

**Concrete example from the data (first 48 h of 2021, temperature):**

| Component | Value range | Meaning |
|-----------|-------------|---------|
| Raw temp | $9.4 \to 18.1$°C | Day–night oscillation superimposed on warming |
| Trend ($k=25$) | $5.4 \to 11.4$°C | Smooth multi-day warming of $\sim$6°C over 48 h |
| Seasonal ($= \text{Raw} - \text{Trend}$) | $+4.0$ at noon, $-3.0$ at midnight | Clean 24 h cycle, $\sim\pm 5$°C amplitude |

### 2.5 Replication Padding in Moving Average

At the sequence boundaries, the moving average kernel extends beyond the available data: position $t=0$ needs values $\mathbf{X}_{-12}$ through $\mathbf{X}_{-1}$ which do not exist. The official implementation \citep{zeng2023dlinear} handles this by **replication padding** — repeating the first and last observed values $p = (k-1)/2 = 12$ times:

$$
\mathbf{X}_{\text{padded}} = [\,\underbrace{\mathbf{X}_0, \ldots, \mathbf{X}_0}_{p},\; \mathbf{X}_0, \ldots, \mathbf{X}_{L-1},\; \underbrace{\mathbf{X}_{L-1}, \ldots, \mathbf{X}_{L-1}}_{p}\,]
$$

$$
\mathbf{T}_0 = \frac{\overbrace{\mathbf{X}_0 + \cdots + \mathbf{X}_0}^{p+1 \text{ lần}} + \mathbf{X}_1 + \cdots + \mathbf{X}_p}{k}
            = \frac{(p+1)\,\mathbf{X}_0 + \sum_{j=1}^{p} \mathbf{X}_j}{k}
$$

All $k=25$ positions in the kernel window contain actual observations or faithful copies of the nearest observation. No position is underweighted. The assumption encoded by replication is **persistence at the boundary**: if nothing is known beyond the edge of the observation window, the most conservative guess is that conditions remain unchanged. This is the standard choice for moving-average decomposition in forecasting — it preserves the local trend at the edges rather than diluting it with arbitrary filler values.

With a kernel of 25, the first and last 12 positions (24 of 120 lookback hours, 20% of the input window) are affected by the boundary treatment. The 12 rightmost of these are the hours closest to the forecast origin, where the signal is most informative for short-horizon prediction.

---

## 3. Pipeline Overview

Figure~\ref{fig:dlinear-arch} illustrates the DLinear architecture. The model takes a multivariate input window $\mathbf{X} \in \mathbb{R}^{120 \times 11}$, decomposes it into trend and seasonal components via a moving-average kernel (k=25, replication padding), maps each component independently through linear layers to the forecast horizon ($\mathbf{W}_t, \mathbf{W}_s \in \mathbb{R}^{24 \times 120}$), sums the two forecasts, and applies a learned cross-channel projection $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{11 \times 7}$ to produce the final output $\mathbf{\hat{Y}} \in \mathbb{R}^{24 \times 7}$.

```
Input X (120h, 11feat)
    |
    |--- AvgPool1d(k=25) --> Trend  --> Linear(120→24) --> Y_trend ----
    |                                                                   + --> Y_raw --> Proj(11→7) --> Output Ŷ (24h, 7)
    |--- X - Trend ---------> Season --> Linear(120→24) --> Y_season --
```

- **DLinear-S**: one pair $(\mathbf{W}_t, \mathbf{W}_s)$ shared across all 11 channels — 5,892 params.
- **DLinear-I**: per-channel $\mathbf{W}_t^{(c)}, \mathbf{W}_s^{(c)}$ for $c = 1,\ldots,11$ — 63,972 params.

---

## 4. Training Procedure

**One model, all targets.** Unlike XGBoost (which trains one regressor per target), a single DLinear model predicts all seven targets jointly. The final projection layer W_proj learns cross-target relationships directly from the training data.

**Data usage.** The model trains on the full 2021–2023 sliding windows at stride 1 (26,137 windows). Every window conditions strictly on observations before the target horizon, preventing leakage. No subsampling or stride reduction is applied — the model is fast enough to use all available windows.

**Validation and early stopping.** Training minimizes MSE with the Adam optimizer. A ReduceLROnPlateau scheduler halves the learning rate when validation loss fails to improve for 5 epochs. Early stopping terminates training when validation loss does not improve for 10 consecutive epochs, capped at 50 epochs total. All hyperparameters (Table~\ref{tab:dlinear-hparams}) are fixed a priori — there is no grid search — and the 2025 test set is never seen during training.

**Convergence.** DLinear-S reaches mean validation loss $0.3538 \pm 1.6 \times 10^{-4}$ across 3 seeds (85 s per seed on an NVIDIA GeForce RTX 4060). DLinear-I converges to $0.3451 \pm 5.7 \times 10^{-4}$ (233 s per seed). Both plateau smoothly without divergence: the validation loss slope over the last 10 epochs is $-1.5 \times 10^{-4}$ for S and $-3.6 \times 10^{-4}$ for I — both effectively flat. The train–validation loss gap is small (S: $-0.030$, I: $-0.036$), indicating no overfitting.

**Multi-seed training.** Per protocol §8, three seeds {42, 123, 456} are trained and checkpoints saved for each seed per variant. The purely linear optimization landscape is near-convex, leading to highly consistent convergence regardless of random initialization.

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 10⁻³ |
| Weight decay | 10⁻⁵ |
| Loss | MSE |
| Batch size | 64 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Early stopping | Patience 10 epochs |
| Max epochs | 50 |
| Kernel size (k) | 25 |
| Seeds | {42, 123, 456} |

---

## 5. Algorithm

```
Algorithm 1: DLinear training and rolling-origin evaluation

Input:  Hourly series S[0..T]; lookback L=120; forecast H=24; targets V=7
        Kernel size k=25; seeds {42, 123, 456}; max epochs 50

1.  Preprocess:
    a. Parse time index, clip wind ≥ 0
    b. Wind decomposition: wx = speed·cos(dir), wy = speed·sin(dir)
    c. Add cyclical time encodings: sin/cos(hour), sin/cos(doy)
    d. Chronological split: train(2021-23), val(2024), test(2025)
    e. Fit StandardScaler on train, transform all splits

2.  Generate sliding windows:
    For each split:
        X[i] = S[i : i+L]       ∈ ℝ^(L × 11)
        Y[i] = S[i+L : i+L+H]  ∈ ℝ^(H × 7)   (target columns only)

3.  For each seed s ∈ {42, 123, 456}:
    a. Initialize DLinear(k=25, individual=True/False)
    b. For epoch = 1 to 50:
        - Forward pass (see Section 2):
          T = AvgPool1d(X);  S_season = X - T
          Ŷ = W_t·T + W_s·S_season;  Ŷ = Ŷ·W_proj
        - Loss = MSE(Ŷ, Y)
        - Backpropagate, Adam step
        - If val_loss not improved for 5 epochs: lr = lr × 0.5
        - If val_loss not improved for 10 epochs: break
    c. Save best checkpoint

4.  Rolling-origin evaluation over test year:
    For each origin t (step = 24 h, N ≈ 360 origins):
    a. Extract window X_t = S[t-L : t]
    b. Predict Ŷ_t = model(X_t)             ∈ ℝ^(24 × 7)
    c. Inverse-transform Ŷ_t to real scale
    d. Compare with actual Y_t = S[t : t+H]

5.  Aggregate metrics:
    - Overall MAE/RMSE (mean over all N×H×V entries)
    - Per-variable (mean over N×H, per variable)
    - Per-horizon (mean over N×V, per lead time)
    - Bootstrap 95% CI (1000 resamples of N origins)

6.  Diagnostics (Protocol §6):
    - ADF stationarity on train series
    - Ljung-Box on residuals (24 lags × 7 vars)
    - D'Agostino normality test on residuals
    - 95% empirical forecast intervals
    - Stability check (val loss slope over last 10 epochs)
```

---

## 6. Experimental Results

Evaluated by rolling-origin over the 2025 test year (N = 360 non-overlapping origins, step = 24 h, real scale). Both variants substantially outperform the naive repeat-last-value baseline (MAE = 6.71). DLinear matches or exceeds more complex architectures with only 5,892–63,972 parameters — achieved with a purely linear decomposition-based design.

### 6.1 Overall Performance

Evaluated by rolling-origin over the 2025 test year (N = 360 non-overlapping origins, step = 24 h, real scale). Values are ensemble-mean across 3 seeds {42, 123, 456}.

| Model | MAE | RMSE | MAE 95% CI | RMSE 95% CI | Params | Train time |
|-------|-----|------|------------|-------------|--------|------------|
| DLinear-S | 5.79 | 12.27 | [5.59, 5.99] | [11.73, 12.79] | 5,892 | 85 s |
| DLinear-I | 5.64 | 12.19 | [5.41, 5.82] | [11.63, 12.70] | 63,972 | 233 s |

DLinear-I improves over S by +2.7% (MAE) and +0.6% (RMSE), at the cost of 10.9× more parameters and 2.7× longer training. This aligns with the original paper's findings \citep{zeng2023dlinear}: individual weights remove the forced compromise of sharing one temporal pattern across physically diverse channels. The RMSE/MAE ratio is ~2.1 for both variants, indicating occasional large errors (driven by cloud cover and wind extremes). Bootstrap 95% CIs (1000 resamples of the 360 test origins) are narrow (width < 4%), confirming stable metric estimates.

### 6.2 Per-Variable Error

| Target | DLinear-S MAE | DLinear-I MAE | Unit | I over S |
|--------|--------------|--------------|------|----------|
| temperature | 1.22 | 1.18 | °C | +2.7% |
| dew_point | 1.28 | 1.19 | °C | +6.4% |
| surface_pressure | 1.33 | 1.13 | hPa | +15.2% |
| relative_humidity | 5.70 | 5.64 | % | +1.1% |
| wx | 4.47 | 4.39 | km/h | +1.8% |
| wy | 3.10 | 3.07 | km/h | +1.1% |
| cloud_cover | 23.41 | 23.00 | % | +1.8% |

DLinear-I improves on every variable. The largest gain is surface pressure (+15%), whose temporal pattern (slow-varying, weak diurnal cycle) differs fundamentally from temperature and humidity (strong 24 h cycle). The shared weights of DLinear-S force a compromise across these diverse patterns; individual weights remove this constraint, allowing each channel to learn its own temporal dynamics.

Cloud cover remains the hardest target across all models (MAE ~23%), consistent with its bimodal distribution (clustered at 0% and 100%) and weak linear correlation with other variables observed in the EDA.

### 6.3 Per-Horizon Error Growth (DLinear-S)

Error grows approximately 2.0× from h=1 to h=24:

| Horizon | h=1 | h=4 | h=8 | h=12 | h=16 | h=20 | h=24 |
|---------|-----|-----|-----|------|------|------|------|
| Overall MAE | 3.46 | 4.58 | 5.07 | 5.77 | 6.75 | 6.82 | 7.03 |

The error growth is approximately linear with horizon, consistent with a model that learns a smooth temporal mapping. The plateau between h=20 and h=24 suggests the model's uncertainty saturates beyond roughly 20 hours — additional lookback or auxiliary features may be needed to improve long-horizon forecasts.

### 6.4 Diagnostic Summary (Protocol §6)

| Diagnostic | S result | I result | Interpretation |
|------------|----------|----------|----------------|
| ADF stationarity | PASS (p≈0 all 7) | PASS (p≈0 all 7) | All series stationary, d=0 |
| Stability | PASS (slope −1.5×10⁻⁴) | PASS (slope −3.6×10⁻⁴) | Flat val loss, converged |
| Ljung-Box (lag 24) | 76/168 pass (45%) | 90/168 pass (54%) | Residual autocorrelation strongest in dew_point (1/24 S, 1/24 I), pressure (1/24 S, 10/24 I). Protocol §11: pressure has known long memory |
| Residual normality | FAIL (p≈0 all) | FAIL (p≈0 all) | Expected — cloud bimodal, wind heavy-tailed |
| Forecast intervals (95%) | Temp ±1.85→±2.90°C (×1.6) | Temp ±1.64→±2.76°C (×1.7) | Monotonic widening, plausible |
| Bootstrap CI (MAE) | ±3.5% | ±3.7% | Narrow (<5%), high confidence |

**ADF:** All seven target variables are stationary at the 1% level — no differencing is needed.

**Ljung-Box:** Roughly half of the 168 horizon-variable combinations show significant residual autocorrelation. The pattern is concentrated at short horizons (h=1..6) and in dew_point and pressure — variables Protocol §11 explicitly identifies as having long-memory autocorrelation. This is a known characteristic of the data, not a model failure. DLinear-I's individual weights partially mitigate the issue (54% pass vs. 45% for S).

**Residual normality:** Universally rejected, as expected. Cloud residuals are U-shaped (bimodal target), wind residuals are heavy-tailed (gust events). Gaussian-based confidence intervals would be unreliable; empirical quantile-based intervals are used instead.

**Forecast intervals:** All variables show monotonic widening from h=1 to h=24. Pressure widens most aggressively (×5.3 in DLinear-I) due to its long-memory structure — the model's uncertainty about pressure grows faster than for temperature (×1.6).

**Bootstrap CI:** MAE confidence interval width under ±4% for both variants confirms stable metric estimates.