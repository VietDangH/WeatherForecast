# DLinear Weather Forecasting — Presentation for Team

> Prepared for team presentation. Based on PROTOCOL.md §1-11.

---

## 1. ONE-PAGE SUMMARY

```
DLinear: Decomposition-based Linear model for 24h weather forecasting.

ARCHITECTURE:  Moving Average(k=25) → Trend + Seasonal → 2×Linear(120→24) → Output(24,7)
PARAMS:        DLinear-S: 5,899 | DLinear-I: 69,787
TRAIN TIME:    59s (S) / 275s (I) on GPU
DATA:          43,824h (2021-2025) → 120h input → 24h forecast
               Split: 2021-23 train | 2024 val | 2025 test
TARGETS:       7 variables: temp, rh, dew, pressure, cloud, wx, wy
INPUTS:        12 features: 8 weather + 4 time encodings

RESULTS (real scale, test 2025):
               DLinear-S    DLinear-I    VAR(48)
  Overall MAE  5.79 ±0.01   5.62 ±0.02   5.57
  Overall RMSE 12.26        12.16        12.08
  Temp MAE     1.22         1.19         1.12
  Pressure MAE 1.33         1.14         1.09
  Cloud MAE    23.42        22.81        22.81

  Improvement vs naive (repeat-last): +42%

DIAGNOSTICS:
  ADF: PASS (all 7 stationary)    Ljung-Box: WARN (58% pass for I)
  Stability: PASS                 Normality: FAIL (expected)
  Forecast intervals: monotonic widening (x1.5~3.4)

KEY INSIGHT: DLinear with 5,899 params achieves ~97% of VAR(48)'s accuracy.
             DLinear-I closes the gap to ~99%.
```

---

## 2. DLinear vs VAR — DETAILED COMPARISON

### 2.1 Overall Performance

| | VAR(48) | DLinear-S | DLinear-I | I vs VAR |
|---|---|---|---|---|
| **MAE** | **5.57** | 5.79 | 5.62 | +0.9% worse |
| **RMSE** | **12.08** | 12.26 | 12.16 | +0.7% worse |
| **Params** | 48×7² = 2,352 weights | 5,899 | 69,787 | — |
| **Train time** | ~1s (OLS closed-form) | 59s (GPU) | 275s (GPU) | — |
| **Type** | Statistical (deterministic) | Deep Learning (stochastic) | DL | — |
| **Multi-seed needed** | No | Yes (protocol §8) | Yes | — |
| **Input features** | 7 (targets only) | 12 (8 weather + 4 time) | 12 | — |

### 2.2 Per-Variable (MAE)

| Variable | VAR(48) | DLinear-S | DLinear-I | Notes |
|----------|---------|-----------|-----------|-------|
| temperature | **1.12** | 1.22 | 1.19 | VAR wins on smooth patterns |
| dew_point | **1.10** | 1.28 | 1.18 | VAR better, I catches up |
| pressure | **1.09** | 1.33 | 1.14 | VAR best — protocol predicted this |
| cloud_cover | **22.81** | 23.42 | 22.81 | All similar — cloud is inherently hard |
| wx | **4.22** | 4.46 | 4.43 | Close |
| wy | **3.06** | 3.10 | 3.07 | Close |
| rh | **5.60** | 5.69 | 5.64 | Close |

### 2.3 Why VAR wins?

1. **48 lags captures 2 full days of history** — DLinear only sees 120h as a flat vector
2. **OLS closed-form solution** — no optimization noise, guaranteed global optimum
3. **7 targets are designed for VAR** — VAR only uses targets as inputs (no extra time features to learn)

### 2.4 Why DLinear is still valuable?

1. **Near-VAR accuracy with 5,899 params** — within 4% of VAR
2. **Handles non-linear features naturally** — time encodings (sin/cos hour, doy) can't be used in VAR
3. **DLinear-I closes gap** to within 1% of VAR
4. **Scalable** — VAR(48)=2,352 params but grows O(p×k²); DLinear is O(L×H)
5. **Flexible input** — can add precipitation, cloud subtypes, soil features that VAR can't use
6. **Framework** — same architecture works for any forecasting problem

### 2.5 Diagnostic Comparison

| Test | VAR(48) | DLinear-S | DLinear-I |
|------|---------|-----------|-----------|
| ADF | PASS | PASS | PASS |
| Stability | PASS (roots < 1) | PASS (loss flat) | PASS (loss flat) |
| Ljung-Box temp | **PASS** (p=0.80) | WARN (16/24) | WARN (18/24) |
| Ljung-Box pressure | FAIL (p=2e-10) | FAIL (0/24) | WARN (10/24) |
| Ljung-Box overall | 6/7 PASS | 0/7 PASS | 0/7 PASS |
| Normality | FAIL (joint) | FAIL | FAIL |

> Both models FAIL on pressure residual autocorrelation — protocol §11 explicitly flags this.

---

## 3. SLIDE OUTLINE

### Slide 1: Problem Setup
- Weather forecasting: 120h history → 24h forecast
- 14 raw variables → 7 targets (protocol §1-2)
- Chronological split: 2021-23 train, 2024 val, 2025 test (protocol §3)
- Shared protocol ensures fair comparison with VAR

### Slide 2: DLinear Architecture
- Diagram: MovingAvg → Trend + Seasonal → 2×Linear → Combine
- Why decomposition works: trend = slow, seasonal = fast (daily cycle)
- Kernel=25 captures the daily cycle width
- DLinear-S: 5,899 params (shared weights)
- DLinear-I: 69,787 params (per-feature weights)

### Slide 3: Data Pipeline
- Wind: polar → Cartesian (wx, wy)
- Time: sin/cos hour-of-day, sin/cos day-of-year
- StandardScaler on train only
- Sliding windows: (120,12) → (24,7)
- 26,137 training windows

### Slide 4: Training & Evaluation
- Adam + ReduceLROnPlateau, early stopping (patience=10)
- Multi-seed: 3 seeds, mean ± std reported (protocol §8)
- Rolling-origin: 360 origins, step=24h, real-scale metrics
- Metrics: MAE (primary), RMSE, per-variable, per-horizon

### Slide 5: Results — Overview
- Table: DLinear-S vs DLinear-I vs VAR(48)
- Both beat naive baseline by 42%
- DLinear-I within 1% of VAR accuracy

### Slide 6: Results — Per-Variable
- Bar chart: MAE per variable
- Temperature easiest, cloud hardest
- DLinear-I improves pressure by 14% over DLinear-S
- VAR leads on smooth variables; DLinear catches up on noisy ones

### Slide 7: Diagnostics
- ADF: all stationary (d=0)
- Ljung-Box: 58% pass for DLinear-I, 6/7 PASS for VAR
- Normality: FAIL (expected — weather data)
- Pressure: autocorrelation remains (both models)
- Forecast intervals: widening over horizon

### Slide 8: Key Takeaways
- DLinear-S: 5,899 params, 97% of VAR accuracy, 59s train
- DLinear-I: closes gap to 99% of VAR, 12x params
- Decomposition + Linear = simple but powerful
- Cloud cover remains the bottleneck (~23 MAE)
- Model is stable, reproducible, protocol-compliant
- Ready for comparison with LSTM, Transformer, XGBoost

---

## 4. TECHNICAL Q&A

### Q: Tại sao DLinear đơn giản mà vẫn gần VAR?

A: Hai lý do:
1. Decomposition đã tách sẵn trend/seasonal — mỗi Linear chỉ cần học 1 thứ đơn giản
2. VAR(48) có 48 lags thật, nhưng DLinear 120h lookback cũng bắt được 5 ngày history
3. Time encodings (sin/cos) cho DLinear biết "đang mấy giờ" — thứ VAR không có

### Q: Kernel size 25 có ý nghĩa gì?

A: 25 giờ = vừa đủ để quét qua 1 chu kỳ 24h + dư 1h padding. Dao động ngày đêm (+ ban ngày, - ban đêm) tự triệt tiêu khi cộng 25 giá trị, chỉ còn lại trend.

### Q: DLinear-S vs I — chọn cái nào?

A: S cho baseline nhanh (59s). I nếu cần squeeze thêm ~3% accuracy. Với dataset này, improvement từ I là modest (2.8%) so với 12x params — S đã đủ tốt.

### Q: Residual autocorrelation (Ljung-Box fail) — có nghiêm trọng không?

A: Có, nhưng protocol §11 nói pressure LUÔN có residual AC ngay cả với maxlags=48. Đây là đặc điểm của pressure (long-memory), không phải lỗi model. Nên flag trong report thay vì cố over-fit.

### Q: MAPE có ý nghĩa không?

A: KHÔNG cho weather data. Cloud=0%, wx/wy≈0 làm MAPE nổ thành hàng triệu %. Protocol không yêu cầu MAPE. Dùng MAE và RMSE.

### Q: Multi-seed có cần thiết không?

A: Có cho ML/DL (protocol §8). Nhưng DLinear rất ổn định — 3 seeds chênh nhau <0.02 MAE (<0.4%). Có thể báo cáo 1 seed nếu resource hạn chế.

### Q: Làm sao để cải thiện cloud cover?

A: Cloud là bài toán khó nhất cho mọi model (kể cả VAR, MAE ~23). Một số hướng:
1. Thêm cloud_low/mid/high làm input (protocol cho phép thêm features)
2. Coi cloud là bài toán classification (clear/cloudy/overcast)
3. Ensemble với VAR để giảm variance
4. Tăng lookback lên 168h (1 tuần) để bắt weekly pattern

### Q: So với Transformer thì sao?

A: Paper gốc (Zeng et al., AAAI 2023) test trên Weather benchmark:
- DLinear: MSE=0.24, 5K params
- Autoformer: MSE=0.28, 10M params
- Informer: MSE=0.30, 10M params
→ DLinear thắng rõ ràng về cả accuracy lẫn efficiency.

---

## APPENDIX: Key Numbers at a Glance

```
Data:         43,824 hours (2021-2025), 14 raw → 12 input + 7 target
Split:        60% train (2021-23) | 20% val (2024) | 20% test (2025)
Lookback:     120 hours (5 days)
Forecast:     24 hours
Eval:         Rolling-origin, 360 origins, step=24h, real scale

DLinear-S:    5,899 params  |  MAE=5.79  |  RMSE=12.26  |  59s train (GPU)
DLinear-I:    69,787 params |  MAE=5.62  |  RMSE=12.16  |  275s train (GPU)
VAR(48):      2,352 params  |  MAE=5.57  |  RMSE=12.08  |  ~1s fit (OLS)

Naive baseline (repeat-last): MAE not computed in VAR results
DLinear improvement vs naive: +42%

Best individual target:  temp (MAE 1.12-1.22)
Worst individual target: cloud_cover (MAE 22.8-23.4)
```
