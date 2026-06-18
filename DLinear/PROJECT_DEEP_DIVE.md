# DLinear Weather Forecasting — Project Deep Dive

> Paper: "Are Transformers Effective for Time Series Forecasting?" — Zeng et al., AAAI 2023
> Protocol: PROTOCOL.md — Shared Experiment Protocol cho nhóm
> Dataset: 43,824 giờ weather data (2021-2025), 14 cột gốc

---

## 1. CƠ SỞ LÝ THUYẾT

### 1.1 Bài toán

Cho 120 giờ quá khứ của 12 biến thời tiết → dự đoán 24 giờ tương lai của 7 biến mục tiêu.

```
Input:  X ∈ R^(120 × 12)    →    Output: Ŷ ∈ R^(24 × 7)
```

### 1.2 DLinear = Decomposition + Linear

DLinear dựa trên 1 insight: **chuỗi thời gian = xu hướng chậm + dao động nhanh**. Tách riêng rồi dự báo từng cái bằng phép nhân ma trận đơn giản.

**Bước 1 — Decomposition (Moving Average, kernel_size = 25):**

```
T_t = (1/25) × Σ_{i=-12}^{12} X_{t+i}      (Trend: xu hướng dài hạn)
S_t = X_t - T_t                              (Seasonal: dao động quanh trend)
```

- **Trend**: đường mượt — nóng dần hay lạnh dần trong 5 ngày qua?
- **Seasonal**: phần còn lại — chu kỳ ngày đêm, mây mưa, nhiễu

Tại sao k=25? Vì 25 giờ đủ để quét qua 1 chu kỳ ngày đêm (24h) + dư 1h → dao động ngày đêm tự triệt tiêu khi cộng dồn, chỉ còn lại xu hướng.

**Bước 2 — Hai nhánh Linear dự báo riêng:**

```
Ŷ_trend    = W_t × Trend    + b_t      (dự báo xu hướng 24h tới)
Ŷ_seasonal = W_s × Seasonal + b_s      (dự báo dao động 24h tới)

Ŷ = Ŷ_trend + Ŷ_seasonal               (dự báo cuối cùng)
```

Mỗi W là ma trận (24 × 120):
- 24 dòng = 24 giờ cần dự báo
- 120 cột = 120 giờ quá khứ
- W[i,j] = "giờ quá khứ thứ j đóng góp bao nhiêu vào giờ tương lai thứ i"

**Bước 3 — Projection (input 12 features → output 7 targets):**

```
Ŷ_final = W_proj × Ŷ        (12 → 7 features, 91 params)
```

### 1.3 Cách model học W

```
Khởi tạo:  W = random nhỏ (gaussian)
Lặp mỗi epoch:
  1. Forward:   Ŷ = model(X)                              (dự đoán)
  2. Loss:      L = (1/N) × Σ(ŷ - y_thật)²               (tính sai)
  3. Backward:  ∂L/∂W                                     (gradient)
  4. Update:    W_mới = W_cũ - learning_rate × gradient   (sửa sai)
  5. Until:     val loss ngừng giảm 10 epochs → early stop
```

**Adam optimizer** cải tiến hơn SGD thuần:
- Momentum: gradient cùng hướng nhiều lần → bước dài hơn (đà)
- Adaptive LR: gradient lớn → giảm LR, gradient nhỏ → tăng LR (tự điều chỉnh)

### 1.4 DLinear-S vs DLinear-I

| | DLinear-S (Shared) | DLinear-I (Individual) |
|---|---|---|
| **W** | 1 cặp cho TẤT CẢ 12 features | 12 cặp, mỗi feature 1 cặp riêng |
| **Params** | 5,899 | 69,787 |
| **Train time** | ~60s (GPU) | ~275s (GPU) |
| **Ưu điểm** | Nhẹ, ít overfit, train nhanh | Mỗi feature học pattern riêng |
| **Nhược điểm** | Features phải "thỏa hiệp" → plateau sớm | Nặng hơn, cần nhiều data hơn |
| **Dùng khi** | Features có pattern giống nhau | Features có pattern khác biệt lớn |

### 1.5 Tại sao DLinear đánh bại Transformer?

Paper chứng minh: **attention trong Transformer thực chất đang học 1 phép tổ hợp tuyến tính có trọng số** giữa các bước thời gian. Linear + decomposition làm đúng việc đó, nhanh hơn, ít params hơn, không cần attention.

```
Weather benchmark:
  DLinear:      5K params, 1 phút train, MSE = 0.24
  Autoformer:  10M params, 2-3 giờ train, MSE = 0.28
  Informer:    10M params, 2-3 giờ train, MSE = 0.30
```

---

## 2. CÁCH DATA ĐƯỢC XỬ LÍ

### 2.1 Pipeline tổng (`prepare_data.py` — theo protocol §2-4)

```
weather_dataset.csv (43,824 dòng × 14 cột)
    │
    ├── [1] Parse time, sort ascending, set index, asfreq('h')
    │
    ├── [2] Sensor-error guard: clip wind columns >= 0
    │       (xử lý artifact -9999 nếu có)
    │
    ├── [3] Wind decomposition (tọa độ cực → Cartesian):
    │       wx = wind_speed × cos(wind_dir_rad)
    │       wy = wind_speed × sin(wind_dir_rad)
    │       (giúp model học được vector gió thay vì góc 0-360°)
    │
    ├── [4] Interpolate gaps (nếu có) + forward/back fill
    │
    ├── [5] Cyclical time encodings (cho model biết "mấy giờ", "mùa nào"):
    │       sin_hour = sin(2π × hour / 24)
    │       cos_hour = cos(2π × hour / 24)
    │       sin_doy  = sin(2π × day_of_year / 365.25)
    │       cos_doy  = cos(2π × day_of_year / 365.25)
    │       (sin/cos đảm bảo 23h và 0h gần nhau về mặt hình học)
    │
    ├── [6] Chronological split (KHÔNG SHUFFLE — protocol §3):
    │       2021-2023 → train  (26,280h, 60%)
    │       2024      → val    (8,784h,  20%)
    │       2025      → test   (8,760h,  20%)
    │
    ├── [7] Extract:
    │       X = 12 input features (8 weather + 4 time)
    │       Y = 7 target variables
    │
    ├── [8] StandardScaler: fit TRÊN TRAIN ONLY, transform val + test
    │       Z = (X - μ_train) / σ_train
    │
    └── Export: outputs/processed_data.npz, scaler.pkl, feature_names.json, target_names.json
```

### 2.2 12 Input Features

| Nhóm | Features | Ghi chú |
|------|----------|---------|
| Weather (8) | temp, rh, dew, precip, pressure, cloud, wx, wy | wx/wy thay thế wind_speed + wind_dir |
| Time (4) | sin_hour, cos_hour, sin_doy, cos_doy | Cyclical encoding |

### 2.3 7 Target Variables

```
temp, rh, dew, pressure, cloud, wx, wy
```

`precip` và 4 time features là **input-only** (không nằm trong targets — protocol không yêu cầu dự báo precipitation).

### 2.4 Sliding Windows (`data_loader.py`)

```
Dữ liệu đã scale:  X_train (26280, 12), Y_train (26280, 7)

Với mỗi window i từ 0 đến (26280 - 120 - 24):
  X[i] = X_train[i : i+120]                  ← 120 giờ quá khứ (5 ngày)
  Y[i] = Y_train[i+120 : i+120+24]           ← 24 giờ tương lai

Kết quả:
  Train: 26,137 windows
  Val:    8,641 windows
  Test:   8,617 windows
```

---

## 3. CÁC THÀNH PHẦN THIẾT KẾ QUAN TRỌNG TRONG CODE

### 3.1 `config.py` — Trái tim của project

Toàn bộ hyperparameters, đường dẫn, constants tập trung ở đây. Không cần sửa code logic — chỉ sửa config.

```python
# === Protocol constants (DO NOT EDIT) ===
TARGET_FEATURES = ['temperature_2m (°C)', ...]   # 7 biến
INPUT_FEATURES  = ['temperature_2m (°C)', ...]   # 12 biến
PROTOCOL_LOOKBACK = 120    # 5 days
PROTOCOL_FORECAST = 24     # 24 hours
PROTOCOL_TRAIN_YEARS = [2021, 2022, 2023]
PROTOCOL_VAL_YEAR = 2024
PROTOCOL_TEST_YEAR = 2025

# === Model config ===
KERNEL_SIZE = 25           # Moving average window
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 10              # Early stopping patience
SEED = 42

# === Path helpers ===
get_checkpoint_dir(model_type)  # → checkpoints/ hoặc checkpoints_i/
get_output_dir(model_type)      # → outputs/S/ hoặc outputs/I/
is_individual(model_type)       # → True nếu "DLinear-I"

# === Protocol self-check ===
_validate_protocol()  # Chạy khi import — AssertionError nếu vi phạm protocol
```

### 3.2 `models/dlinear.py` — Kiến trúc model (222 dòng)

**3 class chính:**

```
MovingAvg(nn.Module):
    # AvgPool1d(kernel_size=25, stride=1, padding=12)
    # Giữ nguyên chiều dài chuỗi nhờ padding
    # Trend[t] = trung bình của 25 điểm xung quanh t

SeriesDecomp(nn.Module):
    # Trend = MovingAvg(X)
    # Seasonal = X - Trend

DLinear(nn.Module):
    __init__(lookback, forecast, n_features, out_features, kernel_size, individual):
        self.decomp = SeriesDecomp(kernel_size)

        if individual:                               # DLinear-I
            self.linear_trend    = ModuleList([Linear(120→24) × 12])
            self.linear_seasonal = ModuleList([Linear(120→24) × 12])
        else:                                        # DLinear-S
            self.linear_trend    = Linear(120→24)    # 1 cái chung
            self.linear_seasonal = Linear(120→24)

        self.proj = Linear(12→7) if n_features != out_features else Identity()

    forward(x):                                      # (B,120,12) → (B,24,7)
        x = x.permute(0,2,1)                        # (B,12,120) cho Conv1d
        trend, seasonal = self.decomp(x)            # tách trend/seasonal

        if individual:
            # Mỗi feature qua Linear riêng
            trend_fc    = stack([linear_i(trend[:,i,:]) for i in 0..11])     # (B,12,24)
            seasonal_fc = stack([linear_i(seasonal[:,i,:]) for i in 0..11])  # (B,12,24)
        else:
            # Tất cả features qua chung 1 Linear
            trend_fc    = linear_trend(trend)            # (B,12,24)
            seasonal_fc = linear_seasonal(seasonal)

        y = trend_fc + seasonal_fc                      # kết hợp 2 nhánh
        y = self.proj(y.permute(0,2,1))                 # (B,24,7)
        return y
```

### 3.3 `train.py` — Training với multi-seed support (260 dòng)

```python
def train_model(model_type="DLinear-S", seeds=None):
    """
    seeds=None  → single seed (dùng config.SEED)
    seeds=list  → multi-seed (tạo checkpoint riêng cho từng seed)
    """
    if seeds is not None:
        return _multi_seed_train(model_type, seeds)
    # ... single-seed training ...

def _multi_seed_train(model_type, seeds):
    """Lặp qua từng seed, train, lưu best_model_seed{seed}.pth"""

# Training loop:
def train_epoch():    # Train 1 epoch trên toàn bộ train_loader
def validate_epoch(): # Validate trên val_loader (no_grad)

# Optimizer: Adam + ReduceLROnPlateau (giảm 50% LR khi val loss plateau 5 epochs)
# Early stopping: dừng khi val loss không cải thiện 10 epochs
# Checkpoint: lưu model tốt nhất theo val loss
```

### 3.4 `evaluate.py` — Rolling-origin evaluation (426 dòng)

```python
# Protocol §7: rolling 24h forecasts, step=24h (non-overlapping)
# 360 origins trên test year 2025

def _evaluate_checkpoint(checkpoint_path):
    """Evaluate 1 checkpoint → metrics trên real scale"""
    1. Load model
    2. Predict trên test_loader, subsample step=24
    3. Denormalize: ŷ_real = ŷ_scaled × σ + μ
    4. Compute: overall MAE/RMSE, per-variable, per-horizon
    5. Bootstrap CI (1000 resamples)
    6. Baseline: repeat-last-value

def _discover_checkpoints():
    """Tìm tất cả checkpoints: S và I, single + multi-seed"""

def evaluate():
    """Duyệt tất cả checkpoints, tổng hợp + so sánh S vs I"""
```

### 3.5 `diagnostics.py` — Chẩn đoán thống kê (427 dòng)

```python
# Protocol §6 — 6 diagnostic tests

1. adf_stationarity_test():
   ADF test trên toàn bộ train → kiểm tra chuỗi có unit root không
   DLinear: n/a (handled by learning) — giữ để biết

2. ljung_box_test():
   Kiểm tra residual autocorrelation tại từng horizon (lag=24)
   H0: white noise. p > 0.05 = PASS
   → nếu fail: model còn bỏ sót pattern

3. residual_normality_check():
   Histogram + Q-Q plot + D'Agostino-Pearson test
   → weather data LUÔN fail (cloud bimodal, wind heavy-tailed)

4. forecast_intervals():
   Empirical quantile-based intervals (95%)
   → widening over horizon — bình thường

5. bootstrap_ci_metrics():
   Resample forecast origins → distribution của MAE/RMSE
   → CI hẹp = kết quả ổn định

6. stability_check():
   Val loss slope last-10 epochs → không phân kỳ?
```

### 3.6 `generate_report.py` — Báo cáo tổng hợp (290 dòng)

```
Tạo diagnostic_report.txt gồm 10 sections:
  1.  Overall Metrics        — MAE, RMSE, RMSE/MAE ratio
  2.  ADF Stationarity       — từng biến: stat, p-value, verdict
  3.  Per-Variable Metrics   — ranked từ dễ → khó
  4.  Per-Horizon Error      — error growth t+1 → t+24
  5.  Ljung-Box Test         — từng biến: pass rate, Q-stat
  6.  Residual Normality     — từng biến: stat, p-value, skewness
  7.  Forecast Intervals     — từng biến: width tại h=1,12,24
  8.  Bootstrap CI           — từng biến: MAE/RMSE ± 95% CI
  9.  Stability Check        — val loss convergence
  10. Final Verdict          — PASS/FAIL + recommendations
```

### 3.7 Import Map (dependency graph)

```
config.py          ← imported by EVERY module (central hub)
models/dlinear.py  ← pure model, no project imports
data_loader.py     ← imports config
prepare_data.py    ← imports config
train.py           ← imports config, data_loader, models.dlinear
evaluate.py        ← imports config, data_loader, models.dlinear, utils
diagnostics.py     ← imports config, data_loader, models.dlinear
generate_report.py ← imports config, data_loader, models.dlinear, diagnostics
utils.py           ← standalone (matplotlib/seaborn), no project imports
```

---

## 4. CÁCH ĐÁNH GIÁ MODEL

### 4.1 Metrics (real scale, sau khi denormalize)

| Metric | Công thức | Dùng cho | Giải thích |
|--------|-----------|----------|------------|
| **MAE** | mean(|ŷ - y|) | **Primary** | Sai số trung bình, dễ hiểu nhất |
| **RMSE** | √mean((ŷ-y)²) | Secondary | Phạt nặng sai số lớn (cloud, wind) |
| **MAPE** | mean(|ŷ-y|/y) × 100% | Info only | % — vô nghĩa khi y ≈ 0 |

### 4.2 Three Breakdowns (protocol §7 yêu cầu)

```
1. Overall       — 1 con số MAE/RMSE cho toàn bộ test set
2. Per-horizon   — sai số tăng thế nào từ t+1 đến t+24
3. Per-variable  — biến nào dễ, biến nào khó
```

### 4.3 Rolling-Origin Evaluation

```
Test year 2025 (8,760 giờ):
  Origin 1:    dùng history đến 00:00 1/1/2025 → dự báo 24h
  Origin 2:    dùng history đến 00:00 2/1/2025 → dự báo 24h
  ...
  Origin 360:  dùng history đến 00:00 26/12/2025 → dự báo 24h

Step = 24h (non-overlapping) → 360 origins
→ KHÔNG dùng predicted values làm input cho bước sau
→ Mỗi origin dùng GROUND TRUTH history
```

### 4.4 Baseline

**Naive (repeat-last-value):** lặp giá trị cuối cùng của mỗi biến 24 lần.

```
ŷ_{t+1} = ŷ_{t+2} = ... = ŷ_{t+24} = x_t
```

Đây là lower bound — model nào cũng phải tốt hơn cái này.

---

## 5. CHẨN ĐOÁN MODEL (PROTOCOL §6)

### 5.1 Tổng quan 6 diagnostic tests

| Test | Hỏi gì | DLinear-S | DLinear-I | Hành động nếu FAIL |
|------|--------|-----------|-----------|---------------------|
| **ADF** | Chuỗi stationary? | PASS (p≈0) | PASS (p≈0) | Differencing |
| **Ljung-Box** | Residual còn pattern? | 80/168 (48%) ⚠️ | 98/168 (58%) ⚠️ | Tăng lookback |
| **Normality** | Residual chuẩn? | FAIL | FAIL | Dùng empirical CI |
| **Stability** | Loss phân kỳ? | PASS | PASS | Giảm LR |
| **Forecast Int** | Độ widening hợp lý? | PASS | PASS | — |
| **Bootstrap CI** | Metrics ổn định? | ±3.7% ✅ | ±3.7% ✅ | — |

### 5.2 Ý nghĩa từng test

**ADF (Augmented Dickey-Fuller):**
- H0: chuỗi có unit root (= không stationary)
- p < 0.05 → bác bỏ H0 → stationary → không cần differencing
- Với DL model: stationary không bắt buộc (model tự học)
- Với VAR: stationary BẮT BUỘC, nếu fail phải differencing

**Ljung-Box:**
- H0: residuals là white noise (không còn pattern)
- Test autocorrelation ở từng horizon (1..24) cho từng biến
- Nếu p < 0.05 tại horizon h → residual tại h còn autocorrelation
- → Model đã bỏ sót pattern ở horizon đó → cần thêm capacity

**Normality (D'Agostino-Pearson):**
- H0: residuals có phân phối chuẩn
- Weather data LUÔN FAIL: cloud bimodal (0% vs 100%), wind heavy-tailed
- Hậu quả: không dùng được Gaussian CI (±1.96σ)
- Giải pháp: dùng empirical quantile intervals

**Forecast Intervals:**
- Dùng empirical quantiles (2.5%, 97.5%) tại mỗi horizon
- Mục đích: biết "dự báo chính xác đến đâu"
- Widening theo horizon là bình thường (càng xa càng không chắc)

**Bootstrap CI:**
- Resample forecast origins với replacement
- Tạo phân phối của MAE/RMSE → tính 95% CI
- CI hẹp (< 5%) = kết quả rất ổn định, không phải may mắn

**Stability:**
- Fit linear regression trên 10 val loss cuối
- Slope > 0 → val loss đang TĂNG → model không ổn định
- Slope ≈ 0 → model đã hội tụ

---

## 6. KẾT QUẢ

### 6.1 DLinear-S vs DLinear-I (real scale, test 2025)

| Target | S MAE | I MAE | Delta |
|--------|-------|-------|-------|
| temperature | 1.22 | 1.19 | +2.3% |
| dew_point | 1.28 | 1.18 | +7.4% |
| surface_pressure | 1.33 | 1.14 | **+14.0%** |
| wy | 3.10 | 3.07 | +0.9% |
| wx | 4.46 | 4.43 | +0.8% |
| rh | 5.70 | 5.64 | +1.2% |
| cloud_cover | 23.42 | 22.81 | +2.6% |
| **OVERALL** | **5.79** | **5.62** | **+2.8%** |

| | S | I |
|---|---|---|
| Params | 5,899 | 69,787 |
| Train time | 59s | 275s |
| Ljung-Box pass | 80/168 | 98/168 |
| Improvement vs naive | +42% | +42% |

### 6.2 Kết luận

- **DLinear-I tốt hơn S ~3%** nhưng tốn 12× params và 5× thời gian
- **Pressure** hưởng lợi nhiều nhất từ I (+14%)
- **Cloud cover** là bottleneck chung (MAE ~23%)
- **Ljung-Box vẫn warning** cho cả 2 — cần thêm lookback
- **Normality FAIL** là bình thường với weather data
- **Cả 2 model đều rất ổn định** (Bootstrap CI < 4%, stability PASS)

---

## 7. QUY TRÌNH CHẠY

```bash
cd DLinear

# 1. Data preparation (chỉ cần chạy 1 lần)
python prepare_data.py

# 2. Train DLinear-S
python -c "from train import train_model; train_model('DLinear-S')"

# 3. Train DLinear-I
python -c "from train import train_model; train_model('DLinear-I')"

# 4. Train multi-seed (protocol §8 — optional)
python -c "from train import train_model; train_model('DLinear-I', seeds=[42,123,456])"

# 5. Evaluate all checkpoints
python evaluate.py

# 6. Diagnostics
python diagnostics.py DLinear-S
python diagnostics.py DLinear-I

# 7. Generate comprehensive reports
python generate_report.py
```

---

## 8. CẤU TRÚC OUTPUTS

```
outputs/
├── processed_data.npz              ← data chung
├── scaler.pkl
├── feature_names.json (12 inputs)
├── target_names.json (7 targets)
│
├── S/                              ← DLinear-S
│   ├── loss_curve.png
│   ├── training_history.json
│   ├── metrics/
│   │   ├── test_metrics.csv
│   │   ├── evaluation_results.json
│   │   ├── per_horizon_error.png
│   │   └── pred_vs_actual/
│   └── diagnostics/
│       ├── diagnostic_report.txt   ← 10-section comprehensive report
│       ├── residual_histograms.png
│       ├── residual_qq_plots.png
│       └── forecast_intervals.png
│
└── I/                              ← DLinear-I
    └── (cấu trúc tương tự)
```
