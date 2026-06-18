# Báo cáo DLinear — Dự báo thời tiết đa biến

> **Mô hình:** DLinear-S (Shared weights)
> **Paper:** *"Are Transformers Effective for Time Series Forecasting?"* — Zeng et al., AAAI 2023
> **Protocol:** Tuân thủ [PROTOCOL.md](PROTOCOL.md) — Weather Forecasting Shared Experiment Protocol
> **Ngày thực hiện:** 11/06/2026

---

## 1. Tổng quan

### 1.1 Bài toán

Dự báo 7 biến thời tiết cho **24 giờ tiếp theo** dựa trên **120 giờ (5 ngày)** dữ liệu quá khứ. Đây là bài toán **time series forecasting đa biến** (multivariate), trong đó mỗi biến đầu ra có mối tương quan với các biến khác và với chính nó trong quá khứ.

### 1.2 Tại sao chọn DLinear?

DLinear là mô hình **tuyến tính có cấu trúc** (structured linear model), được công bố tại AAAI 2023. Paper gốc chứng minh rằng một mô hình tuyến tính đơn giản nhưng được thiết kế tốt có thể **vượt qua các mô hình Transformer phức tạp** trên nhiều benchmark time series, bao gồm cả dữ liệu thời tiết.

**Ý tưởng cốt lõi:** Thay vì dùng attention mechanism phức tạp, DLinear phân rã chuỗi thời gian thành 2 thành phần — **xu hướng (trend)** và **dao động (seasonal)** — rồi dự báo từng thành phần bằng một lớp Linear đơn giản. Cách tiếp cận này vừa nhanh vừa hiệu quả, đặc biệt khi dữ liệu có tính chu kỳ rõ rệt như thời tiết.

---

## 2. Dữ liệu & Tiền xử lý

### 2.1 Dataset

| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | `weather_dataset.csv` (Open-Meteo) |
| Thời gian | 01/01/2021 – 31/12/2025 (5 năm) |
| Tần suất | 1 giờ (hourly) |
| Số dòng | 43,824 |
| Số cột gốc | 14 (1 time + 13 numeric) |
| Dữ liệu khuyết | Không có |

**13 cột dữ liệu gốc:**

| # | Cột | Đơn vị | Mô tả |
|---|-----|--------|-------|
| 1 | temperature_2m | °C | Nhiệt độ không khí ở độ cao 2m |
| 2 | relative_humidity_2m | % | Độ ẩm tương đối |
| 3 | dew_point_2m | °C | Điểm sương |
| 4 | precipitation | mm | Lượng mưa |
| 5 | surface_pressure | hPa | Áp suất khí quyển bề mặt |
| 6 | cloud_cover | % | Tỷ lệ mây che phủ tổng |
| 7 | cloud_cover_low | % | Mây tầng thấp |
| 8 | cloud_cover_mid | % | Mây tầng trung |
| 9 | cloud_cover_high | % | Mây tầng cao |
| 10 | wind_speed_10m | km/h | Tốc độ gió ở độ cao 10m |
| 11 | wind_direction_10m | ° (0–360) | Hướng gió (góc) |
| 12 | wind_gusts_10m | km/h | Gió giật |
| 13 | soil_moisture_0_to_7cm | m³/m³ | Độ ẩm đất tầng 0–7cm |

### 2.2 Pipeline tiền xử lý (theo PROTOCOL.md §2)

Pipeline gồm 5 bước, thực hiện trong `DLinear/prepare_data.py`:

#### Bước 1: Load, sắp xếp, đặt tần suất giờ
```python
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').set_index('time')
df = df.asfreq('h')   # đảm bảo mỗi dòng cách nhau đúng 1h
```
→ Đảm bảo dữ liệu theo đúng thứ tự thời gian, index là timestamp.

#### Bước 2: Sửa lỗi sensor — clip tốc độ gió ≥ 0
```python
df['wind_speed_10m (km/h)'] = df['wind_speed_10m (km/h)'].clip(lower=0)
df['wind_gusts_10m (km/h)'] = df['wind_gusts_10m (km/h)'].clip(lower=0)
```
→ Tốc độ gió không thể âm về mặt vật lý. Kết quả: 0 giá trị âm được phát hiện — dữ liệu sạch.

#### Bước 3: Phân rã hướng gió → tọa độ Cartesian (wx, wy)
```python
wind_dir_rad = np.deg2rad(df['wind_direction_10m (°)'])
df['wx'] = df['wind_speed_10m (km/h)'] * np.cos(wind_dir_rad)   # Đông (+wx)
df['wy'] = df['wind_speed_10m (km/h)'] * np.sin(wind_dir_rad)   # Bắc (+wy)
```

**Tại sao cần bước này?** Hướng gió là biến vòng (circular): 1° và 359° cách nhau 2° về mặt hình học nhưng cách 358 đơn vị về số học nếu dùng trực tiếp. Phân rã thành wx, wy giải quyết vấn đề này, đồng thời **giữ được cả thông tin tốc độ VÀ hướng** trong 2 biến liên tục.

#### Bước 4: Điền dữ liệu khuyết (nếu có)
```python
df = df.interpolate(method='time').ffill().bfill()
```
→ Dữ liệu hiện tại không có null, nhưng bước này vẫn chạy để đảm bảo an toàn cho các lần chạy sau.

#### Bước 5: Thêm 4 đặc trưng thời gian chu kỳ (ML/DL only)
```python
df['sin_hour'] = np.sin(2π × hour / 24)     # chu kỳ ngày (sin)
df['cos_hour'] = np.cos(2π × hour / 24)     # chu kỳ ngày (cos)
df['sin_doy']  = np.sin(2π × dayofyear / 365.25)  # chu kỳ năm (sin)
df['cos_doy']  = np.cos(2π × dayofyear / 365.25)  # chu kỳ năm (cos)
```
→ Mã hóa vị trí trong chu kỳ ngày và chu kỳ năm. Dùng sin/cos để 23h và 0h nằm gần nhau trên đường tròn đơn vị, không bị "đứt" như khi dùng số thô 0–23.

### 2.3 Chọn đặc trưng đầu vào (12) và đầu ra (7)

Dựa trên yêu cầu của PROTOCOL.md §2, mô hình DLinear (thuộc nhóm ML/DL) sử dụng:

**Input — 12 đặc trưng** (8 thời tiết + 4 thời gian):

| # | Đặc trưng | Nguồn | Vai trò |
|---|----------|--------|---------|
| 0 | temperature_2m (°C) | CSV gốc | Nhiệt độ — biến quan trọng nhất |
| 1 | relative_humidity_2m (%) | CSV gốc | Độ ẩm |
| 2 | dew_point_2m (°C) | CSV gốc | Điểm sương (tương quan cao với temp) |
| 3 | precipitation (mm) | CSV gốc | Lượng mưa (chỉ dùng làm input, không dự báo) |
| 4 | surface_pressure (hPa) | CSV gốc | Áp suất bề mặt |
| 5 | cloud_cover (%) | CSV gốc | Mây tổng |
| 6 | wx | Tính từ gió | Thành phần gió Đông–Tây |
| 7 | wy | Tính từ gió | Thành phần gió Bắc–Nam |
| 8 | sin_hour | Tính từ time | Mã hóa giờ trong ngày |
| 9 | cos_hour | Tính từ time | Mã hóa giờ trong ngày |
| 10 | sin_doy | Tính từ time | Mã hóa ngày trong năm |
| 11 | cos_doy | Tính từ time | Mã hóa ngày trong năm |

**Target — 7 biến:**

| # | Biến | Đơn vị | Ghi chú |
|---|------|--------|---------|
| 0 | temperature_2m | °C | Quan trọng nhất |
| 1 | relative_humidity_2m | % | Biến động theo mùa |
| 2 | dew_point_2m | °C | Tương quan mạnh với temp |
| 3 | surface_pressure | hPa | Ít biến động |
| 4 | cloud_cover | % | Phân bố lưỡng cực (0% hoặc 100%) |
| 5 | wx | km/h | Thành phần gió Đông–Tây |
| 6 | wy | km/h | Thành phần gió Bắc–Nam |

**Các cột bị loại khỏi model:**

| Cột bị loại | Lý do |
|------------|-------|
| cloud_cover_low/mid/high | Đã có `cloud_cover` tổng — 3 cột con dư thừa |
| wind_direction_10m | Đã phân rã thành wx, wy |
| wind_gusts_10m | Nhiễu cao, không làm input hiệu quả |
| soil_moisture_0_to_7cm | Chỉ có 1 tầng đất, thiếu tầng sâu để có ý nghĩa |
| precipitation (target) | Rất nhiễu (nhiều giá trị 0), chỉ dùng làm input |

### 2.4 Phân chia dữ liệu (PROTOCOL.md §3)

Dữ liệu được chia theo năm, **tuyệt đối không xáo trộn** (no shuffle) để tránh rò rỉ thông tin từ tương lai:

| Tập | Năm | Số giờ | Tỷ lệ | Vai trò |
|-----|-----|--------|-------|---------|
| **Train** | 2021, 2022, 2023 | 26,280 | 60% | Huấn luyện tham số mô hình |
| **Validation** | 2024 | 8,784 | 20% | Early stopping & chọn hyperparameter |
| **Test** | 2025 | 8,760 | 20% | Đánh giá cuối cùng (không đụng vào khi train) |

**Lý do split theo năm:** Mỗi năm chứa đủ 4 mùa, đảm bảo cả 3 tập đều có đầy đủ tính chất thời tiết theo mùa. Nếu split ngẫu nhiên, model có thể vô tình học từ các ngày liền kề trong test set.

### 2.5 Chuẩn hóa (PROTOCOL.md §4)

Sử dụng `StandardScaler` — công thức: **z = (x − μ) / σ**

```python
scaler = StandardScaler()
scaler.fit(all_train)           # Fit CHỈ trên train
all_val_scaled  = scaler.transform(all_val)    # Dùng scaler từ train
all_test_scaled = scaler.transform(all_test)   # Dùng scaler từ train
```

**Nguyên tắc quan trọng:** Scaler chỉ được fit trên tập train. Validation và test dùng lại scaler đã fit từ train — mô phỏng thực tế: không được biết trước dữ liệu tương lai. Tất cả metrics được tính trên **giá trị thực (real scale)** sau khi denormalize.

### 2.6 Cửa sổ trượt (Sliding Windows)

Từ dữ liệu đã scale, tạo các cửa sổ trượt:

```
Input  X: 120 giờ quá khứ → shape (batch, 120, 12)
Target Y:  24 giờ tương lai → shape (batch,  24,  7)

Số cửa sổ tạo được:
  Train: 26,137   (26,280 − 120 − 24 + 1)
  Val:    8,641
  Test:   8,617
```

---

## 3. Kiến trúc mô hình DLinear

### 3.1 Toán học cơ bản

DLinear dựa trên phép phân rã chuỗi thời gian cổ điển:

$$\mathbf{X} = \mathbf{T} + \mathbf{S}$$

Trong đó:
- **Trend (T):** Thành phần xu hướng — được trích xuất bằng **Moving Average** với kernel size `k = 25`
- **Seasonal (S):** Phần dao động = X − T

**Công thức toán:**

$$\mathbf{T} = \text{AvgPool1d}(\mathbf{X}, \text{kernel}=k)$$
$$\mathbf{S} = \mathbf{X} - \mathbf{T}$$
$$\hat{\mathbf{Y}}_{\text{trend}} = W_t \cdot \mathbf{T} + b_t$$
$$\hat{\mathbf{Y}}_{\text{seasonal}} = W_s \cdot \mathbf{S} + b_s$$
$$\hat{\mathbf{Y}} = \hat{\mathbf{Y}}_{\text{trend}} + \hat{\mathbf{Y}}_{\text{seasonal}}$$

Với $W_t, W_s \in \mathbb{R}^{H \times L}$ — mỗi ma trận ánh xạ từ 120 bước quá khứ → 24 bước tương lai. Trong DLinear-S (Shared), **tất cả 12 kênh (features) dùng chung một cặp W_t, W_s**.

### 3.2 Forward pass (mã giả)

```
Input:  X = (batch, 120, 12)

1. X → permute → (batch, 12, 120)              // chuẩn bị cho Conv1d
2. T = AvgPool1d(kernel=25, padding=12)(X)      // trích trend
3. S = X - T                                     // seasonal = residual
4.   Branch Trend:   T → Linear(120→24) → Ŷ_trend   (batch, 12, 24)
5.   Branch Seasonal: S → Linear(120→24) → Ŷ_seasonal (batch, 12, 24)
6. Combine: Ŷ_trend + Ŷ_seasonal → (batch, 12, 24)
7. Project: Linear(12→7) → (batch, 7, 24)
8. Permute → Output: (batch, 24, 7)
```

### 3.3 DLinear-S vs DLinear-I

| | DLinear-S (đang dùng) | DLinear-I |
|---|---|---|
| **Cơ chế** | 1 cặp Linear cho tất cả channels | Mỗi channel có cặp Linear riêng |
| **Số tham số** | 2 × 120 × 24 + 12 × 7 = **5,844** | 2 × 12 × 120 × 24 + 12 × 7 = **69,204** |
| **Khi dùng** | Features có pattern tương tự nhau | Features có pattern khác biệt |
| **Rủi ro** | Underfit nếu features quá khác biệt | Overfit nếu ít dữ liệu |

Mô hình hiện tại dùng **DLinear-S** với **5,844 tham số** — cực kỳ nhẹ so với các mô hình deep learning thông thường (LSTM/Transformer thường có hàng trăm nghìn đến hàng triệu tham số). Điều này giúp train nhanh, ít overfit, nhưng có thể thiếu capacity để nắm bắt hết các pattern phức tạp.

### Kết quả huấn luyện thực tế: S vs I

| Metric | DLinear-S (5,844 params) | DLinear-I (69,204 params) |
|--------|--------------------------|---------------------------|
| Val loss (seed 42) | 0.354157 | **0.350109** |
| Thời gian train (3 seeds) | ~2.5 phút | ~5 phút |
| Overfit gap (train-val) | ~9% | ~10% |

DLinear-I hội tụ với val loss thấp hơn, nhưng gap train-val rộng hơn một chút (dấu hiệu overfit nhẹ). Đây là điều mong đợi với 12× tham số.

### 3.4 Hàm mất mát

Sử dụng **Mean Squared Error (MSE):**

$$\mathcal{L} = \frac{1}{N \cdot H \cdot C} \sum_{i=1}^{N} \sum_{j=1}^{H} \sum_{c=1}^{C} (\hat{y}_{i,j,c} - y_{i,j,c})^2$$

Tối ưu bằng **Adam optimizer** với learning rate khởi tạo `1e-3`, kèm **ReduceLROnPlateau** (giảm LR xuống 50% nếu val loss không cải thiện sau 5 epoch).

---

## 4. Huấn luyện

### 4.1 Cấu hình

| Tham số | Giá trị | Giải thích |
|---------|--------|------------|
| MODEL_TYPE | DLinear-S | Shared weights cho tất cả channels |
| KERNEL_SIZE | 25 | Kích thước cửa sổ moving average |
| LOOKBACK | 120 giờ | 5 ngày quá khứ |
| FORECAST | 24 giờ | 1 ngày dự báo |
| BATCH_SIZE | 64 | Số mẫu mỗi batch |
| LEARNING_RATE | 1e-3 | Tốc độ học khởi tạo |
| EPOCHS | 50 | Số epoch tối đa |
| PATIENCE | 10 | Dừng sớm nếu val loss không cải thiện sau 10 epoch |
| WEIGHT_DECAY | 1e-5 | L2 regularization (nhẹ) |
| OPTIMIZER | Adam | Adaptive gradient descent |
| SCHEDULER | ReduceLROnPlateau | Giảm LR ×0.5 sau 5 epoch không cải thiện |
| SEED | 42 | Khởi tạo ngẫu nhiên cố định |
| DEVICE | CUDA (GPU) | Tăng tốc huấn luyện |

### 4.2 Diễn biến huấn luyện

```
Epoch   Train Loss      Val Loss        LR          Patience
==============================================================
1       0.577553        0.414766        1.00e-03    0/10
2       0.353082        0.365716        1.00e-03    0/10
3       0.331282        0.359171        1.00e-03    0/10
...
12      0.324236        0.356008        1.00e-03    1/10
13      0.323310        0.355248        5.00e-04    2/10      ← LR giảm
20      0.323273        0.354157        5.00e-04    5/10      ← BEST ←
...
27      0.322940        0.354490        2.50e-04    6/10      ← LR giảm lần 2
30      0.322912        0.354843        2.50e-04    9/10
Early stop @ epoch 30
```

**Nhận xét:**

- **Hội tụ nhanh:** Loss giảm mạnh trong 2 epoch đầu (0.58 → 0.35), sau đó plateau.
- **Khoảng cách train-val nhỏ:** Train loss ≈ 0.323, val loss ≈ 0.354, chênh lệch ~9% — dấu hiệu **chưa overfit nặng**.
- **Model tốt nhất:** Epoch 20 với val_loss = 0.354157.
- **LR giảm 2 lần:** Từ 1e-3 → 5e-4 (epoch 13) → 2.5e-4 (epoch 27), cho thấy model đã bão hòa.
- **Early stop ở epoch 30:** Patience = 10 được kích hoạt — val loss không cải thiện trong 10 epoch cuối.
- **Thời gian train:** ~52 giây trên GPU — cực kỳ nhanh nhờ kiến trúc tuyến tính đơn giản.

### 4.3 Multi-seed training (PROTOCOL.md §8)

Để đánh giá tính ổn định của mô hình (ML/DL là stochastic), DLinear được huấn luyện với 3 seeds khác nhau:

| Seed | Best Epoch | Val Loss |
|------|-----------|----------|
| 42 | 20 | 0.354157 |
| 123 | 22 | 0.354512 |
| 456 | 18 | 0.354690 |

Kết quả nhất quán giữa các seeds (val loss chênh lệch < 0.2%) — DLinear rất ổn định nhờ kiến trúc đơn giản.

---

## 5. Kết quả đánh giá

### 5.1 Phương pháp đánh giá (PROTOCOL.md §7)

Đánh giá theo phương pháp **rolling-origin** với bước 24 giờ (không chồng lấn):
- Với mỗi điểm gốc `t` trong năm 2025, dùng 120 giờ lịch sử để dự báo 24 giờ tiếp theo
- Bước nhảy giữa các lần dự báo: **24 giờ** (phủ toàn bộ năm 2025, không chồng lấn)
- Tổng số lần dự báo: **360 origins** (= floor(8,617 / 24))

Tất cả metrics được tính trên **giá trị thực (real scale)** sau khi denormalize qua StandardScaler.

### 5.2 Kết quả tổng thể (Multi-Seed: Mean ± Std)

| Chỉ số | DLinear (mean ± std) | Naive Baseline | Cải thiện |
|--------|---------------------|-----------------|-----------|
| **MAE** | **5.786 ± 0.014** | 6.713 | **+13.8%** |
| **RMSE** | **12.262 ± 0.004** | 16.048 | **+23.6%** |
| MAE 95% CI | [5.586, 5.994] | — | — |
| RMSE 95% CI | [11.725, 12.794] | — | — |

> *Naive baseline = lặp lại giá trị cuối cùng của 120h lịch sử cho toàn bộ 24h dự báo. Đây là lower bound — bất kỳ model ML nào cũng phải vượt qua baseline này.*

**Ý nghĩa:** Trung bình trên tất cả 7 biến và 24 giờ, DLinear dự đoán sai lệch **5.79 đơn vị** (tính theo MAE). Model cải thiện **~14%** so với việc chỉ lặp giá trị cuối cùng — chứng tỏ model có học được pattern từ dữ liệu quá khứ, nhưng mức cải thiện ở mức khiêm tốn (kỳ vọng 30–50% với model tốt). Multi-seed cho thấy kết quả **rất ổn định** (σ < 0.25% MAE).

### 5.3 Kết quả theo từng biến (Per-Variable)

| Biến | MAE | RMSE | MAPE | Đánh giá |
|------|-----|------|------|----------|
| **temperature_2m (°C)** | 1.22 | 1.66 | 5.6% | ★★★★ Dự đoán tốt nhất — sai số ~1.2°C là chấp nhận được cho dự báo 24h |
| **dew_point_2m (°C)** | 1.28 | 1.87 | 12.1% | ★★★★ Rất tốt — tương quan mạnh với nhiệt độ |
| **surface_pressure (hPa)** | 1.33 | 1.88 | 0.1% | ★★★★ Rất tốt — áp suất ít biến động, dễ dự đoán |
| **wy (km/h)** | 3.10 | 4.32 | huge | ★★★ Khá — thành phần gió Bắc–Nam |
| **wx (km/h)** | 4.46 | 6.04 | huge | ★★★ Khá — thành phần gió Đông–Tây (khó hơn wy) |
| **relative_humidity_2m (%)** | 5.70 | 7.73 | 8.1% | ★★ Trung bình — độ ẩm biến động phức tạp |
| **cloud_cover (%)** | 23.42 | 30.47 | huge | ★ Kém — mây có phân bố lưỡng cực, rất khó dự đoán |

**Phân tích:**
- **Biến dễ (temp, dew, pressure):** Đây là các biến có tính chu kỳ rõ rệt (ngày/đêm, mùa), ít nhiễu. DLinear nắm bắt tốt các pattern này nhờ cơ chế decomposition.
- **Biến khó (cloud, rh):** Mây và độ ẩm biến động thất thường, phân bố không chuẩn. DLinear-S với shared weights gặp khó khăn vì các biến này có pattern rất khác biệt so với temp/pressure.
- **Gió (wx, wy):** Kết quả ở mức trung bình. wx (Đông–Tây) khó hơn wy (Bắc–Nam), phù hợp với đặc điểm khí hậu Việt Nam (gió mùa Bắc–Nam chiếm ưu thế).
- **MAPE cao với cloud, wx, wy** là do các biến này thường xuyên có giá trị gần 0, khiến phép chia cho giá trị thực cho kết quả rất lớn. MAPE không phù hợp cho các biến có phân bố gần 0.

### 5.4 Kết quả theo từng giờ (Per-Horizon)

| Giờ dự báo | MAE | RMSE | Nhận xét |
|-----------|-----|------|----------|
| **1h** | 3.46 | 7.72 | Dự đoán gần nhất — sai số thấp nhất |
| **6h** | 4.76 | 10.89 | Sai số tăng ~38% so với giờ 1 |
| **12h** | 5.79 | 11.81 | Sai số tăng ~67% |
| **18h** | 7.03 | 14.37 | Sai số tăng ~103% |
| **24h** | 7.04 | 14.10 | Xa nhất — sai số cao nhất, tăng ~103% so với giờ 1 |

![Per-Horizon Error](outputs/per_horizon_error.png)

**Phân tích:**
- **Sai số tích lũy theo thời gian:** MAE tăng từ 3.46 (giờ 1) lên 7.04 (giờ 24), tương đương **tăng 103%**.
- **Đường cong mượt:** Không có bước nhảy đột ngột — model hoạt động ổn định, không có hiện tượng divergence.
- **Dự báo ngắn hạn tốt:** Trong 6 giờ đầu, MAE ở mức 3.5–4.8, đủ chính xác cho các ứng dụng cảnh báo thời tiết ngắn hạn.
- **Plateau ở cuối horizon (h=18–24):** Sai số không tăng nhiều sau giờ 18, cho thấy model đã đạt trạng thái cân bằng — dự đoán xa hơn cũng không tệ hơn đáng kể.

### 5.5 Predicted-vs-Actual Plots

Biểu đồ so sánh giá trị dự đoán và thực tế cho từng biến được lưu trong `outputs/pred_vs_actual/`. Mỗi biến được vẽ ở 4 thời điểm khác nhau trong năm 2025 (Q1, Q2, Q3, Q4) để đánh giá chất lượng dự báo theo mùa.

---

## 6. Chẩn đoán mô hình (PROTOCOL.md §6)

### 6.1 ADF Stationarity Test (trên toàn bộ train, real scale)

**Cải tiến phương pháp:** ADF test được chạy trên **toàn bộ 26,280 giờ train** (real scale, đã denormalize), thay vì chỉ 64 mẫu từ một batch. Điều này đảm bảo test có đủ statistical power và sử dụng đúng critical values của phân phối ADF.

| Biến | ADF Statistic | p-value | Kết luận |
|------|--------------|---------|----------|
| temperature_2m | −6.39 | 0.0000 | stationary |
| relative_humidity_2m | −12.36 | 0.0000 | stationary |
| dew_point_2m | −7.11 | 0.0000 | stationary |
| surface_pressure | −7.06 | 0.0000 | stationary |
| cloud_cover | −14.97 | 0.0000 | stationary |
| wx | −13.05 | 0.0000 | stationary |
| wy | −13.70 | 0.0000 | stationary |

> **Kết luận:** Tất cả 7 chuỗi đều dừng (stationary) với độ tin cậy cao (p < 0.0001). Với VAR, không cần differencing (d=0). Với DL, tính dừng được xử lý tự động bởi cơ chế học — không cần can thiệp.

### 6.2 Ljung-Box Test — Per-Horizon Residual Autocorrelation

**Cải tiến phương pháp:** Thay vì pool toàn bộ 200K+ residuals vào một test duy nhất (dẫn đến Q-statistic hàng triệu và p=0.0000 cho mọi biến), test được chạy **riêng cho từng forecast horizon**. Mỗi test sử dụng N=360 residuals (một residual cho mỗi origin, tại horizon đó). Điều này cho kết quả **có giá trị chẩn đoán** thay vì blanket reject.

**H0:** Phần dư tại horizon h là nhiễu trắng. p > 0.05 = PASS.

| Biến | Worst h | Q-stat | p-value | Passed | Đánh giá |
|------|---------|--------|---------|--------|----------|
| **wy** | 15 | 56.18 | 0.0002 | **19/24** | Tốt nhất — ít autocorrelation nhất |
| **cloud_cover** | 18 | 44.96 | 0.0059 | **17/24** | Tốt — mây khó dự đoán nhưng phần dư gần nhiễu trắng |
| **temperature_2m** | 2 | 418.49 | 0.0000 | **16/24** | Khá — các horizon xa (h>10) pass tốt |
| **wx** | 20 | 42.88 | 0.0103 | **14/24** | Trung bình |
| **relative_humidity_2m** | 1 | 51.75 | 0.0008 | **13/24** | Trung bình — horizon gần fail nhiều |
| **dew_point_2m** | 2 | 1063.05 | 0.0000 | **1/24** | Kém — shared weights không tách được dew khỏi temp |
| **surface_pressure** | 2 | 708.75 | 0.0000 | **0/24** | Kém nhất — pressure có cấu trúc riêng, cần individual weights |

**Tổng:** 80/168 horizon-variable pairs PASS (47.6%).

#### DLinear-I Ljung-Box: Cải thiện sau khi dùng individual weights

Với DLinear-I (mỗi channel có W riêng), Ljung-Box cải thiện đáng kể ở các biến cốt lõi:

| Biến | DLinear-S | DLinear-I | Thay đổi |
|------|-----------|-----------|----------|
| **surface_pressure** | **0/24** | **10/24** | **+10** ← khẳng định giả thuyết: pressure cần W riêng |
| **relative_humidity** | 13/24 | **19/24** | +6 |
| **cloud_cover** | 17/24 | **20/24** | +3 |
| **temperature** | 16/24 | **18/24** | +2 |
| **dew_point** | 1/24 | **3/24** | +2 (vẫn yếu) |
| **wx** | 14/24 | **15/24** | +1 |
| **wy** | 19/24 | 13/24 | −6 (nhiễu) |
| **Tổng** | **80/168** | **98/168** | **+18** |

**Kết luận:** DLinear-I giải quyết được vấn đề của pressure (0→10) nhưng chưa đủ cho dew_point (1→3). Dew_point có autocorrelation ~0.99, cần lookback dài hơn (336h) bên cạnh individual weights.

**Phân tích:**
- **Near-horizon autocorrelation (h=1–6):** Hầu hết các biến fail ở horizon gần. Đây là hiện tượng tự nhiên — residuals ở các horizon liền kề có tương quan do dùng chung input window. Không phải là lỗi model.
- **pressure (0/24) và dew_point (1/24):** Fail toàn bộ hoặc gần như toàn bộ. Đây là bằng chứng rõ ràng rằng DLinear-S **không đủ capacity** cho các biến có pattern chậm/persistence cao. DLinear-I được khuyến nghị.
- **wy (19/24) và cloud (17/24):** Kết quả tốt bất ngờ — residuals của các biến nhiễu cao gần với white noise nhất, vì model không học được gì từ chúng (dự đoán ≈ trung bình).

### 6.3 Residual Normality Test

**H0:** Phần dư có phân phối chuẩn. p > 0.05 = PASS.

| Biến | D'Agostino Stat | p-value | Kết luận |
|------|-----------------|---------|----------|
| temperature_2m | 1112 | 0.0000 | FAIL |
| relative_humidity_2m | 264 | 0.0000 | FAIL |
| dew_point_2m | 1734 | 0.0000 | FAIL |
| surface_pressure | 1189 | 0.0000 | FAIL |
| cloud_cover | 307 | 0.0000 | FAIL |
| wx | 447 | 0.0000 | FAIL |
| wy | 581 | 0.0000 | FAIL |

> **Kết luận:** Phần dư không chuẩn — đây là điều **bình thường** với dữ liệu thời tiết (phân bố đuôi dày, bất đối xứng). Không ảnh hưởng nghiêm trọng đến chất lượng dự báo điểm. Cloud có phân bố lưỡng cực (0% hoặc 100%) nên residual không thể chuẩn.

Biểu đồ histogram và Q-Q plot được lưu tại `outputs/residual_histograms.png` và `outputs/residual_qq_plots.png`.

### 6.4 Forecast Intervals (Khoảng dự báo 95%)

Dùng empirical residual quantiles tại mỗi horizon (conformal-style prediction intervals):

| Biến | h=1 (±) | h=12 (±) | h=24 (±) | Độ mở rộng (×) |
|------|---------|----------|----------|----------------|
| temperature_2m | ±1.93°C | ±2.32°C | ±2.84°C | 1.5× |
| relative_humidity_2m | ±5.51% | ±15.20% | ±18.46% | 3.3× |
| dew_point_2m | ±2.05°C | ±4.06°C | ±5.37°C | 2.6× |
| surface_pressure | ±2.02 hPa | ±4.24 hPa | ±5.19 hPa | 2.6× |

> **Phân tích:** Khoảng bất định mở rộng theo horizon — từ ±1.93°C ở giờ 1 lên ±2.84°C ở giờ 24 (nhiệt độ), phù hợp với lý thuyết. Độ ẩm có khoảng rất rộng (±18.5% ở h=24) do biến động cao. Đây là thông tin quan trọng cho người dùng cuối: "Ngày mai nhiệt độ khoảng 25°C, nhưng có thể dao động ±2.8°C."

### 6.5 Stability Check

```
Val loss last-10 slope = −0.000174 → OK (không diverging)
```

Model ổn định, loss không có xu hướng tăng — xác nhận không bị divergence.

### 6.6 Bootstrap CI trên Metrics

| Metric | Giá trị trung bình | 95% CI |
|--------|-------------------|--------|
| MAE | 5.76 | [5.56, 5.98] |
| RMSE | 12.24 | [11.72, 12.79] |

Khoảng tin cậy tương đối hẹp — kết quả ổn định, không phụ thuộc vào mẫu cụ thể.

### 6.7 Tổng hợp chẩn đoán (Diagnostic Verdict)

| Diagnostic | Kết quả | Hành động |
|-----------|---------|-----------|
| ADF Stationarity | **PASS** (7/7 stationary) | d=0 cho VAR, không cần differencing |
| Ljung-Box | **PARTIAL** (80/168 pass) | Tăng capacity (DLinear-I) cho pressure và dew_point |
| Normality | FAIL | Bình thường với weather data |
| Forecast Intervals | OK (widening đúng) | Có thể dùng để báo cáo bất định |
| Bootstrap CI | OK (CI hẹp) | Kết quả ổn định |
| Stability | OK | Không divergence |

> **Kết luận chính:** So với lần chạy trước (ADF sai do dùng 64 mẫu scaled, Ljung-Box blanket FAIL do pool 200K residuals), kết quả chẩn đoán hiện tại có ý nghĩa thống kê. Tất cả chuỗi đều dừng — đúng với kỳ vọng. Ljung-Box xác nhận DLinear-S bỏ sót tín hiệu ở **surface_pressure** và **dew_point** — đây là các biến cần DLinear-I để mô hình hóa riêng. Các biến còn lại ở mức chấp nhận được.

---

## 7. So sánh với Baseline

| Phương pháp | MAE | RMSE | Cải thiện MAE |
|------------|-----|------|---------------|
| **Naive (repeat last)** | 6.713 | 16.048 | — |
| **DLinear-S (seed 123)** | 5.789 | 12.265 | **+13.8%** |
| **DLinear-S (multi-seed mean)** | 5.786 ± 0.014 | 12.262 ± 0.004 | **+13.8%** |

DLinear vượt baseline một cách có ý nghĩa thống kê (bootstrap CI của DLinear [5.59, 5.99] không chồng lấn với naive MAE 6.71). Tuy nhiên, mức cải thiện ~14% là **khiêm tốn** — các mô hình phức tạp hơn thường đạt 30–50%. Điều này phù hợp với chẩn đoán Ljung-Box: model còn yếu ở một số biến, cần nâng cấp lên DLinear-I.

### 7.1 So sánh DLinear-S vs DLinear-I

DLinear-I (Individual weights, 69,204 params) được huấn luyện để so sánh với DLinear-S (Shared weights, 5,844 params):

| Chỉ số | DLinear-S (mean ± std) | DLinear-I (mean ± std) | Cải thiện |
|--------|-----------------------|-----------------------|-----------|
| **MAE** | 5.7864 ± 0.0141 | **5.6241 ± 0.0190** | **+2.8%** |
| **RMSE** | 12.2619 ± 0.0044 | **12.1581 ± 0.0143** | **+0.8%** |

**Cải thiện theo từng biến (MAE):**

| Biến | DLinear-S | DLinear-I | Cải thiện | Kết luận |
|------|-----------|-----------|-----------|----------|
| **surface_pressure** | 1.3264 | **1.1407** | **+14.0%** | ★★★★ Ấn tượng — individual weights giải phóng pressure khỏi shared weights ràng buộc |
| **dew_point_2m** | 1.2796 | **1.1849** | **+7.4%** | ★★★ Khá — còn room để cải thiện thêm |
| **cloud_cover** | 23.4232 | **22.8131** | **+2.6%** | ★★ Nhẹ — cloud vẫn là biến khó nhất |
| **temperature_2m** | 1.2227 | **1.1949** | **+2.3%** | ★★ Nhẹ — temp vốn đã tốt |
| **relative_humidity** | 5.7030 | **5.6371** | **+1.2%** | ★ Nhẹ |
| **wx** | 4.4619 | **4.4283** | **+0.8%** | — Gần như không đổi |
| **wy** | 3.1032 | **3.0741** | **+0.9%** | — Gần như không đổi |

**Cải thiện Ljung-Box (pass rate):**

| Biến | DLinear-S | DLinear-I | Thay đổi |
|------|-----------|-----------|----------|
| **surface_pressure** | **0/24** | **10/24** | **+10** ← DLinear-I thực sự học được pressure dynamics |
| **relative_humidity** | 13/24 | **19/24** | +6 |
| **cloud_cover** | 17/24 | **20/24** | +3 |
| **temperature** | 16/24 | **18/24** | +2 |
| **dew_point** | 1/24 | **3/24** | +2 |
| **wx** | 14/24 | **15/24** | +1 |
| **wy** | 19/24 | 13/24 | −6 (bất thường) |
| **Tổng** | **80/168** | **98/168** | **+18** |

**Phân tích:**
- **Giả thuyết được xác nhận:** Pressure và dew_point cần individual weights. DLinear-I cải thiện pressure MAE 14% và Ljung-Box từ 0→10/24. Đây là các biến có cấu trúc chuỗi thời gian hoàn toàn khác biệt so với temp — không thể dùng chung W.
- **Các biến còn lại ít thay đổi:** Temp, rh, cloud, wx, wy vốn đã có Ljung-Box pass rate khá với DLinear-S. Wy giảm nhẹ (19→13/24), có thể do overfit nhẹ ở I (69K params → dễ học nhiễu).
- **DLinear-I chưa đủ cho dew_point:** Chỉ 3/24 pass, dù MAE cải thiện 7.4%. Dew_point có autocorrelation rất cao (~0.99), cần lookback dài hơn (336h) chứ không chỉ individual weights.
- **Trade-off:** DLinear-I cải thiện pressure rõ rệt nhưng các biến wind (wx, wy) hơi giảm — có thể do overfit cục bộ ở 12 channels riêng lẻ.

---

## 8. Kết luận & Hướng cải thiện

### 8.1 Tổng kết

| Khía cạnh | Đánh giá |
|----------|----------|
| **Pipeline** | Hoàn chỉnh, tuân thủ đầy đủ PROTOCOL.md. Bugs đã được sửa (rolling-origin, ADF, Ljung-Box) |
| **Tốc độ train** | S: ~52 giây, I: ~2 phút (30 epochs trên GPU) |
| **Độ chính xác (S)** | MAE 5.786 ± 0.014, cải thiện 13.8% so với baseline |
| **Độ chính xác (I)** | MAE 5.624 ± 0.019, cải thiện 16.2% so với baseline (+2.8% so với S) |
| **Tính ổn định** | Rất tốt — σ < 0.25% MAE qua 3 seeds, CI hẹp, loss không divergence |
| **Điểm mạnh (S)** | Dự báo tốt temp (1.22°C), dew point (1.28°C), pressure (1.33 hPa) |
| **Điểm yếu (S)** | Cloud (MAE 23.4%), humidity (MAE 5.7%), Ljung-Box fail ở pressure 0/24 và dew_point 1/24 |
| **Cải thiện (I)** | Pressure MAE -14%, Ljung-Box pressure 0→10/24, dew_point MAE -7.4% |

### 8.2 Kết quả đã đạt được

| Biện pháp | Dự đoán | Kết quả thực tế |
|-----------|---------|-----------------|
| **DLinear-I** (individual weights) | Pressure và dew_point cải thiện nhiều nhất | ✅ **Pressure MAE -14%, Ljung-Box 0→10/24** = giả thuyết xác nhận |
| Wind (wx, wy) | Không thay đổi nhiều | ✅ Đúng — wind vốn là white noise |
| Cloud, humidity, temp | Cải thiện nhẹ | ✅ Đúng — +1-3% |
| Dew_point | 1→3/24 Ljung-Box (cần thêm lookback) | ⚠️ Đúng — individual weights chưa đủ, residual còn autocorrelation mạnh |

### 8.3 Hướng cải thiện (theo thứ tự ưu tiên)

1. **Tăng LOOKBACK lên 336h (14 ngày):** DLinear-I đã cải thiện pressure nhưng dew_point chỉ 3/24 Ljung-Box pass. Dew_point có autocorrelation ~0.99 (nhớ rất dài), cần ít nhất 336h lookback. Đây là hướng khả thi nhất để cải thiện thêm.

2. **Thử DLinear-I + lookback=336:** Kết hợp cả individual weights và lookback dài. Kỳ vọng: pressure → 15/24 Ljung-Box, dew_point → 10/24.

3. **Thử kernel_size khác:** Hiện tại k=25. Nên thử k=7 (ngày), k=24 (khớp chu kỳ ngày), k=168 (tuần). Kernel size ảnh hưởng trực tiếp đến chất lượng phân rã trend/seasonal.

4. **Thêm forecast intervals vào báo cáo:** Khoảng dự báo (VD: "nhiệt độ 25°C ± 2.8°C") có giá trị thực tiễn cao hơn là chỉ báo điểm.

5. **Xem xét mô hình khác:** Nếu DLinear-I + lookback=336 vẫn không cải thiện đáng kể dew_point, thử các kiến trúc mạnh hơn như PatchTST, iTransformer, hoặc đơn giản là XGBoost/LightGBM với feature engineering tốt.

---

## Phụ lục A: Cấu trúc project

```
DLinear/
├── PROTOCOL.md              ← Protocol chuẩn cho mọi model
├── config.py                ← Toàn bộ hyperparameter (có auto-validate)
├── prepare_data.py          ← Pipeline tiền xử lý (§2-4)
├── data_loader.py           ← Load dữ liệu → PyTorch DataLoader
├── models/
│   └── dlinear.py           ← DLinear model (S & I variants)
├── train.py                 ← Training loop + multi-seed (§8) + train_i()
├── evaluate.py              ← Rolling-origin evaluation (§7) + S vs I comparison
├── diagnostics.py           ← 6 diagnostic methods (§6) — per-horizon Ljung-Box
├── utils.py                 ← Visualization helpers
├── checkpoints/             ← DLinear-S checkpoints (3 seeds)
│   ├── best_model.pth
│   ├── best_model_seed42.pth
│   ├── best_model_seed123.pth
│   └── best_model_seed456.pth
├── checkpoints_i/           ← DLinear-I checkpoints (3 seeds)
│   ├── best_model_seed42.pth
│   ├── best_model_seed123.pth
│   └── best_model_seed456.pth
└── outputs/
    ├── processed_data.npz   ← Dữ liệu đã scale
    ├── scaler.pkl           ← StandardScaler object
    ├── feature_names.json   ← 12 input names
    ├── target_names.json    ← 7 target names
    ├── loss_curve.png       ← Biểu đồ loss (S)
    ├── loss_curve_I.png     ← Biểu đồ loss (I)
    ├── training_history.json← Lịch sử loss (S)
    ├── training_history_I.json ← Lịch sử loss (I)
    ├── test_metrics.csv     ← Metrics per variable
    ├── evaluation_results.json ← Kết quả đầy đủ (incl. multi-seed, S vs I)
    ├── per_horizon_error.png   ← Biểu đồ sai số theo giờ
    ├── pred_vs_actual/      ← Biểu đồ predicted-vs-actual (§8)
    ├── forecast_intervals.png  ← Khoảng dự báo
    ├── residual_histograms.png ← Histogram phần dư
    └── residual_qq_plots.png   ← Q-Q plot phần dư
```

## Phụ lục B: Các lệnh chạy

```bash
cd DLinear

# 1. Chuẩn bị dữ liệu
python prepare_data.py

# 2. Huấn luyện DLinear-S (đơn seed)
python train.py

# 3. Huấn luyện DLinear-S (đa seed — §8)
python -c "from train import multi_seed_train; multi_seed_train()"

# 4. Huấn luyện DLinear-I (đơn seed)
python -c "from train import train_i; train_i()"

# 5. Huấn luyện DLinear-I (đa seed + so sánh)
python -c "from train import multi_seed_train_i; multi_seed_train_i()"

# 6. Đánh giá (tự động detect S và I checkpoints, so sánh)
python evaluate.py

# 7. Chẩn đoán (6 methods, per-horizon Ljung-Box)
python diagnostics.py
```

## Phụ lục C: Bug fixes & Cải tiến

### Bug fixes (11/06/2026)

| Bug | File | Mô tả | Impact |
|-----|------|-------|--------|
| Rolling-origin subsampling | evaluate.py | Subsampling trong batch (step=24) không duy trì step nhất quán qua batch boundaries | 404 origins → 360 origins (đúng) |
| ADF test | diagnostics.py | Chỉ dùng 64 mẫu scaled từ val_loader thay vì toàn bộ train real-scale | "Non-stationary" → "Stationary" (đúng) |
| Ljung-Box pooling | diagnostics.py | Pool 200K+ residuals → Q=1.4M, p=0.0000 cho mọi biến | Per-horizon: 80/168 pass (có giá trị chẩn đoán) |
| Residual denormalization | diagnostics.py | Cộng mean shift vào residuals (mean không cancel) | Giá trị residual chính xác |
| plot_prediction_vs_actual | utils.py | Gọi scaler.inverse_transform với (24,7) nhưng scaler có 19 columns | Function hoạt động (dùng real-scale input) |
| Multi-seed evaluation | evaluate.py | Chỉ đọc best_model.pth, bỏ qua các seed checkpoints | Auto-detect và aggregate multi-seed |

### Cải tiến (12/06/2026)

| Tính năng | File | Mô tả |
|-----------|------|-------|
| DLinear-I training | train.py | Thêm `train_model(model_type)`, `train_i()`, `multi_seed_train_i()` |
| Checkpoint phân tách | config.py | `CHECKPOINT_DIR_I` + `get_checkpoint_dir()`, `is_individual()` |
| S vs I comparison | evaluate.py | Tự động detect checkpoints từ 2 thư mục, so sánh MAE/RMSE per-variable |
| Checkpoint metadata | train.py | Lưu `model_type` trong checkpoint → evaluate tự chọn individual flag |

---

*Báo cáo được tạo bởi DLinear pipeline, tuân thủ PROTOCOL.md v2.0. Mọi số liệu được tính trên tập Test 2025 với 360 non-overlapping rolling origins × 24h forecasts. Đã so sánh DLinear-S (5,844 params) vs DLinear-I (69,204 params), 3 seeds mỗi loại.*
