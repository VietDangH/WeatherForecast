# Plan Chi Tiết: Áp dụng DLinear cho Weather Forecasting

> **Paper gốc:** "Are Transformers Effective for Time Series Forecasting?" — Zeng et al., AAAI 2023
> **Dataset:** 43,824 dòng hourly weather data (2021–2025), 20 features, không missing

---

## Mục lục

1. [Cơ sở lý thuyết & Toán học](#1-cơ-sở-lý-thuyết--toán-học)
2. [Thuật toán chi tiết](#2-thuật-toán-chi-tiết)
3. [Cấu trúc project](#3-cấu-trúc-project)
4. [Implement chi tiết từng file](#4-implement-chi-tiết-từng-file)
5. [Metric đánh giá](#5-metric-đánh-giá)
6. [Biểu đồ & Trực quan hóa](#6-biểu-đồ--trực-quan-hóa)
7. [Script chạy & Độ chính xác](#7-script-chạy--độ-chính-xác)
8. [Verification Checklist](#8-verification-checklist)

---

## 1. Cơ sở lý thuyết & Toán học

### 1.1 Bài toán Time Series Forecasting

Cho chuỗi thời gian đa biến $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_L] \in \mathbb{R}^{L \times C}$, với:
- $L$: độ dài cửa sổ lịch sử (lookback window)
- $C$: số lượng biến (channels/features)
- $\mathbf{x}_t \in \mathbb{R}^C$: vector giá trị của tất cả biến tại thời điểm $t$

**Mục tiêu:** Dự đoán $H$ bước tiếp theo $\hat{\mathbf{Y}} = [\hat{\mathbf{y}}_{L+1}, ..., \hat{\mathbf{y}}_{L+H}] \in \mathbb{R}^{H \times C}$ sao cho gần nhất với giá trị thực $\mathbf{Y}$.

### 1.2 Phân rã chuỗi thời gian (Time Series Decomposition)

Một chuỗi thời gian cổ điển được phân rã thành 3 thành phần:

$$\mathbf{X} = \mathbf{T} + \mathbf{S} + \mathbf{R}$$

Trong đó:
- **Trend** ($\mathbf{T}$): Xu hướng dài hạn, chuyển động chậm, mượt
- **Seasonal** ($\mathbf{S}$): Dao động có chu kỳ (ngày, tuần, năm)
- **Residual** ($\mathbf{R}$): Phần dư/nhiễu không giải thích được

**Mô hình DLinear đơn giản hóa:** Gom Seasonal và Residual thành một thành phần duy nhất:

$$\mathbf{X} = \mathbf{T} + \mathbf{S}' \quad \text{với} \quad \mathbf{S}' = \mathbf{S} + \mathbf{R}$$

### 1.3 Trích xuất Trend bằng Moving Average

Trend được tính bằng phép Moving Average (MA) với kernel size $k$:

$$\mathbf{T}_t = \frac{1}{k} \sum_{i=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} \mathbf{X}_{t+i}$$

Hoặc viết dưới dạng ma trận: $\mathbf{T} = \mathbf{A} \cdot \mathbf{X}$, trong đó $\mathbf{A} \in \mathbb{R}^{L \times L}$ là ma trận trung bình trượt (Toeplitz matrix với các phần tử $1/k$ trên $k$ đường chéo).

**Trong PyTorch:** Sử dụng `nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)` để tính $\mathbf{T}$.

### 1.4 DLinear — Mô hình tuyến tính kép

Sau khi phân rã, mỗi thành phần được dự báo riêng bằng **một lớp Linear duy nhất**:

**Trend branch:**
$$\hat{\mathbf{Y}}_t = \mathbf{W}_t \cdot \mathbf{T} + \mathbf{b}_t$$

**Seasonal branch:**
$$\hat{\mathbf{Y}}_s = \mathbf{W}_s \cdot \mathbf{S}' + \mathbf{b}_s$$

**Final prediction:**
$$\hat{\mathbf{Y}} = \hat{\mathbf{Y}}_t + \hat{\mathbf{Y}}_s$$

Với $\mathbf{W}_t, \mathbf{W}_s \in \mathbb{R}^{H \times L}$ (với DLinear-S) hoặc $\mathbf{W}_t^{(c)}, \mathbf{W}_s^{(c)} \in \mathbb{R}^{H \times L}$ cho từng channel $c$ (với DLinear-I).

### 1.5 Tại sao DLinear hoạt động? — Góc nhìn Đại số tuyến tính

Kết hợp hai nhánh, ta có:

$$\hat{\mathbf{Y}} = \mathbf{W}_s \cdot \mathbf{X} + (\mathbf{W}_t - \mathbf{W}_s) \cdot \mathbf{T}$$

Vì $\mathbf{T} = \mathbf{A} \cdot \mathbf{X}$ (MA là toán tử tuyến tính):

$$\hat{\mathbf{Y}} = \mathbf{W}_s \cdot \mathbf{X} + (\mathbf{W}_t - \mathbf{W}_s) \cdot \mathbf{A} \cdot \mathbf{X}$$

$$\hat{\mathbf{Y}} = [\mathbf{W}_s + (\mathbf{W}_t - \mathbf{W}_s)\mathbf{A}] \cdot \mathbf{X}$$

Đây là **một phép biến đổi tuyến tính duy nhất** của $\mathbf{X}$, nhưng được cấu trúc hóa: nó tách biệt trọng số cho xu hướng mượt ($\mathbf{A}\mathbf{X}$) và phần dao động ($\mathbf{X} - \mathbf{A}\mathbf{X}$). Việc tách biệt này giúp mô hình học hiệu quả hơn so với một Linear layer đơn thuần.

### 1.6 Hàm mất mát (Loss Function)

**Mean Squared Error (MSE):**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N \cdot H \cdot C} \sum_{i=1}^{N} \sum_{j=1}^{H} \sum_{c=1}^{C} \left(\hat{y}_{i,j,c} - y_{i,j,c}\right)^2$$

Trong đó $N$ là batch size.

**Gradient descent với Adam optimizer:**

$$\theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Với $\alpha$ = learning rate, $\hat{m}_t$ và $\hat{v}_t$ là bias-corrected 1st/2nd moment estimates.

### 1.7 DLinear-S vs DLinear-I

| | DLinear-S (Shared) | DLinear-I (Individual) |
|---|---|---|
| **Tham số** | $2 \times L \times H$ | $2 \times C \times L \times H$ |
| **Ví dụ (L=336, H=96)** | ~64K | ~1.3M (C=20) |
| **Cơ chế** | 1 bộ Linear cho tất cả channels | Mỗi channel có Linear riêng |
| **Khi dùng** | Features có pattern giống nhau | Features có pattern khác biệt |
| **Rủi ro** | Underfit nếu features quá khác biệt | Overfit nếu ít dữ liệu |

---

## 2. Thuật toán chi tiết

### 2.1 Pseudocode — DLinear Forward Pass

```
Algorithm: DLinear Forward Pass
─────────────────────────────────────────────
Input:  X ∈ R^{batch × L × C}       # Historical window
        kernel_size ∈ Z⁺             # Moving avg kernel
Output: Ŷ ∈ R^{batch × H × C}       # Forecast

1: X ← X.transpose(1, 2)            # → (batch, C, L) for Conv1d
2: 
3: # Decomposition via Moving Average
4: T ← AvgPool1d(X, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
5: S ← X - T                         # Seasonal = Original - Trend
6:
7: # Reshape for Linear layer
8: T ← T.transpose(1, 2)            # → (batch, L, C)
9: S ← S.transpose(1, 2)            # → (batch, L, C)
10:
11: # Branch 1: Trend forecasting
12: Ŷ_trend ← Linear_trend(T.transpose(1, 2))  # Linear maps L → H per channel
13: # → (batch, C, H)
14:
15: # Branch 2: Seasonal forecasting
16: Ŷ_seasonal ← Linear_seasonal(S.transpose(1, 2))
17: # → (batch, C, H)
18:
19: # Combine and reshape
20: Ŷ ← (Ŷ_trend + Ŷ_seasonal).transpose(1, 2)
21: # → (batch, H, C)
22:
23: return Ŷ
```

### 2.2 Pseudocode — Training Loop

```
Algorithm: Training Loop
─────────────────────────────────────────────
Input:  train_loader, val_loader, model, config
Output: Trained model, loss history

1: Initialize optimizer = Adam(model.parameters(), lr=config.LR)
2: Initialize criterion = MSELoss()
3: Initialize best_val_loss = ∞, patience_counter = 0
4: Initialize history = {train_loss: [], val_loss: []}
5:
6: for epoch = 1 to config.EPOCHS:
7:     # ── TRAINING PHASE ──
8:     model.train()
9:     train_loss_sum = 0
10:     for X_batch, Y_batch in train_loader:
11:         Ŷ = model(X_batch)
12:         loss = criterion(Ŷ, Y_batch)
13:         optimizer.zero_grad()
14:         loss.backward()
15:         optimizer.step()
16:         train_loss_sum += loss.item()
17:     
18:     avg_train_loss = train_loss_sum / len(train_loader)
19:     history.train_loss.append(avg_train_loss)
20:
21:     # ── VALIDATION PHASE ──
22:     model.eval()
23:     val_loss_sum = 0
24:     with torch.no_grad():
25:         for X_batch, Y_batch in val_loader:
26:             Ŷ = model(X_batch)
27:             loss = criterion(Ŷ, Y_batch)
28:             val_loss_sum += loss.item()
29:     
30:     avg_val_loss = val_loss_sum / len(val_loader)
31:     history.val_loss.append(avg_val_loss)
32:
33:     # ── LOGGING ──
34:     print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
35:
36:     # ── CHECKPOINT & EARLY STOPPING ──
37:     if avg_val_loss < best_val_loss:
38:         best_val_loss = avg_val_loss
39:         patience_counter = 0
40:         Save checkpoint(model, optimizer, epoch, best_val_loss)
41:     else:
42:         patience_counter += 1
43:         if patience_counter >= config.PATIENCE:
44:             print(f"Early stopping at epoch {epoch}")
45:             break
46:
47: return model, history
```

### 2.3 Pseudocode — Sliding Window Data Construction

```
Algorithm: Create Sliding Windows
─────────────────────────────────────────────
Input:  data ∈ R^{T × C}          # All time series data
        L: lookback window        # e.g., 336
        H: forecast horizon       # e.g., 96
Output: X ∈ R^{N × L × C}, Y ∈ R^{N × H × C}

1: N ← T - L - H + 1              # Number of windows
2: Initialize X[N, L, C], Y[N, H, C]
3:
4: for i = 0 to N-1:
5:     X[i] ← data[i : i+L]       # Past window
6:     Y[i] ← data[i+L : i+L+H]   # Future window
7:
8: return X, Y
```

---

## 3. Cấu trúc project

```
e:\WeatherForecast\
├── data\
│   └── weather_dataset.csv              (có sẵn — 43,824 dòng, 21 cột)
│
├── eda.ipynb                            ★ TRUNG TÂM DATA PROCESSING:
│                                         │  - EDA hiện có (cells 0-11)
│                                         │  - Feature Selection (cell mới)
│                                         │  - Feature Engineering (cell mới)
│                                         │  - Data Split & Scale (cell mới)
│                                         │  - Export processed data → outputs/
│
├── plan\
│   └── dlinear_implementation_plan.md   (file này)
│
├── src\
│   ├── __init__.py
│   ├── config.py                        # TẤT CẢ hyperparameter tập trung ở đây
│   ├── data_loader.py                   # Load processed data từ outputs/ → sliding windows → DataLoader
│   ├── models\
│   │   ├── __init__.py
│   │   └── dlinear.py                   # DLinear-S & DLinear-I PyTorch modules
│   ├── train.py                         # Training loop + validation + early stopping
│   ├── evaluate.py                      # Test evaluation + toàn bộ metrics
│   └── utils.py                         # Visualization, metrics functions
│
├── checkpoints\                         # Lưu model .pth
├── outputs\                             # ★ Đầu ra từ eda.ipynb & training
│   ├── processed_data.npz               #     Dữ liệu đã scale & split (từ notebook)
│   ├── scaler.pkl                       #     Scaler object (từ notebook)
│   ├── feature_names.json               #     Danh sách features (từ notebook)
│   ├── loss_curve.png                   #     Biểu đồ loss (từ train.py)
│   ├── test_metrics.csv                 #     Metrics per feature (từ evaluate.py)
│   └── training_history.json           #     Loss history (từ train.py)
│
├── requirements.txt
└── README.md
```

> **★ Nguyên tắc đồng bộ:** `eda.ipynb` chịu trách nhiệm TOÀN BỘ data processing (đọc CSV → chọn features → encode cyclic → split → scale). Kết quả được export ra `outputs/`. Các script `.py` trong `src/` CHỈ load dữ liệu đã xử lý từ `outputs/`, không tự ý xử lý lại từ CSV gốc.

---

## 4. Implement chi tiết từng file

### 4.1 `requirements.txt`

```
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

---

### 4.2 ★ `eda.ipynb` — TRUNG TÂM DATA PROCESSING

Đây là file QUAN TRỌNG NHẤT cho data pipeline. Giữ nguyên 12 cells EDA hiện có (cells 0-11), **bổ sung** các cells mới bên dưới để xử lý data cho DLinear.

#### Cell 12: Feature Selection

Dựa trên kết quả EDA (correlation heatmap, boxplot, phân phối), chọn features đầu vào cho DLinear.

```python
# ────────────────────────────────────────────────────────
# CELL 12: FEATURE SELECTION
# ────────────────────────────────────────────────────────
# Dựa trên EDA (cells 1-11), quyết định chọn/loại features

import pandas as pd
import numpy as np

# Load lại data (đảm bảo chạy độc lập)
df = pd.read_csv("data/weather_dataset.csv")
df['time'] = pd.to_datetime(df['time'])

# ── LÝ DO CHỌN / LOẠI TỪNG FEATURE ────────────────────

selected_features = [
    # === NHIỆT ĐỘ (3 features) ===
    'temperature_2m (°C)',          # ★ Target chính, phân phối chuẩn, chu kỳ rõ
    'dew_point_2m (°C)',            # Tương quan cao với temp (~0.9), pattern mượt
    'apparent_temperature (°C)',    # Tương quan cực cao với temp (~0.98)
    # → Giữ cả 3 vì DLinear-S shared weights học được mối quan hệ tuyến tính
    
    # === ĐỘ ẨM (1 feature) ===
    'relative_humidity_2m (%)',     # Biến động theo mùa, tương quan nghịch với temp (~-0.6)
    
    # === LƯỢNG MƯA (1 feature) ===
    'precipitation (mm)',           # Rất nhiễu, nhiều 0, khó dự đoán nhất — để kiểm tra giới hạn model
    
    # === ÁP SUẤT (2 features) ===
    'pressure_msl (hPa)',           # Tương quan cao với surface_pressure (~0.99)
    'surface_pressure (hPa)',       # → Giữ cả 2: có thể bỏ 1 nếu muốn giảm chiều
    
    # === MÂY (4 features) ===
    'cloud_cover (%)',              # Phân bố lưỡng cực (0% hoặc 100%)
    'cloud_cover_low (%)',
    'cloud_cover_mid (%)',
    'cloud_cover_high (%)',         # Có thể bỏ nếu dùng cloud_cover tổng
    
    # === GIÓ (5 features → 7 sau encode) ===
    'wind_speed_10m (km/h)',        # Tương quan cao với wind_speed_100m (~0.95)
    'wind_speed_100m (km/h)',
    'wind_direction_10m (°)',       # ★ Sẽ encode thành sin/cos trong cell 13
    'wind_direction_100m (°)',      # ★ Sẽ encode thành sin/cos trong cell 13
    'wind_gusts_10m (km/h)',        # Biến động mạnh, pattern tương tự wind_speed
    
    # === ĐẤT (4 features) ===
    'soil_temperature_0_to_7cm (°C)',     # Tương quan ~1.0 với soil_temp_7_28cm
    'soil_temperature_7_to_28cm (°C)',    # → Có thể bỏ 1 nếu muốn giảm chiều
    'soil_moisture_0_to_7cm (m³/m³)',    # Biến đổi chậm, dễ dự đoán
    'soil_moisture_7_to_28cm (m³/m³)',   # Tương quan cao với moisture 0-7cm (~0.98)
]

print(f"Selected: {len(selected_features)} features")

# ── CÁC FEATURES BỊ LOẠI & LÝ DO ─────────────────────
excluded_features = [c for c in df.columns if c not in selected_features and c != 'time']
print(f"\nExcluded features ({len(excluded_features)}):")
for f in excluded_features:
    print(f"  ✗ {f}")
    # (Không có feature nào bị loại trong lần đầu — giữ tất cả 20 features)
    # Nếu muốn giảm, có thể bỏ 1 trong 2 soil_temp, 1 trong 2 soil_moisture, cloud_cover_low/mid/high
```

#### Cell 13: Feature Engineering (Cyclic Encoding)

```python
# ────────────────────────────────────────────────────────
# CELL 13: FEATURE ENGINEERING
# ────────────────────────────────────────────────────────
# Encode wind direction (0°–360°) → sin/cos

# Sao chép DataFrame đã chọn features
df_model = df[['time'] + selected_features].copy()

# ── Encode wind direction ──────────────────────────────
# Toán học: 1° và 359° cách nhau 2° về mặt hình học,
# nhưng cách 358 đơn vị về mặt số học → sin/cos giải quyết vấn đề này

for dir_col in ['wind_direction_10m (°)', 'wind_direction_100m (°)']:
    if dir_col in df_model.columns:
        radians = np.deg2rad(df_model[dir_col])
        base_name = dir_col.replace(' (°)', '').replace('wind_direction', 'wind_dir')
        df_model[f'{base_name}_sin'] = np.sin(radians)
        df_model[f'{base_name}_cos'] = np.cos(radians)
        df_model = df_model.drop(columns=[dir_col])
        print(f"  ✓ {dir_col} → {base_name}_sin, {base_name}_cos")

# ── Kết quả ────────────────────────────────────────────
feature_cols = [c for c in df_model.columns if c != 'time']
n_features = len(feature_cols)
print(f"\nFeatures after encoding: {n_features}")

# Hiển thị để verify
print("\nDanh sách features cuối cùng:")
for i, col in enumerate(feature_cols):
    print(f"  [{i:2d}] {col}")

# ── Save feature names ─────────────────────────────────
import json
os.makedirs('outputs', exist_ok=True)
with open('outputs/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print("\n✓ Saved feature_names.json")
```

#### Cell 14: Chronological Split & Scaling

```python
# ────────────────────────────────────────────────────────
# CELL 14: DATA SPLIT & SCALING
# ────────────────────────────────────────────────────────
# QUAN TRỌNG: Split theo thời gian (chronological), KHÔNG shuffle

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# ── Trích xuất numpy array ─────────────────────────────
time_values = df_model['time'].values
feature_values = df_model[feature_cols].values.astype(np.float32)

print(f"Data shape: {feature_values.shape}")

# ── Chronological split: 70/15/15 ─────────────────────
T = len(feature_values)
train_end = int(T * 0.70)
val_end = int(T * 0.85)  # 70 + 15

train_data = feature_values[:train_end]
val_data = feature_values[train_end:val_end]
test_data = feature_values[val_end:]

print(f"\nSplit summary:")
print(f"  Train: {train_data.shape[0]:,} rows | {time_values[0]} → {time_values[train_end-1]}")
print(f"  Val:   {val_data.shape[0]:,} rows | {time_values[train_end]} → {time_values[val_end-1]}")
print(f"  Test:  {test_data.shape[0]:,} rows | {time_values[val_end]} → {time_values[-1]}")

# ── Verify no overlap ──────────────────────────────────
assert train_end + (val_end - train_end) + (T - val_end) == T
print(f"\n✓ Split integrity check passed (total = {T:,})")

# ── StandardScaler (fit on TRAIN only) ─────────────────
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# Verify scaling
print(f"\nScaling verification (train only):")
print(f"  Mean: {train_scaled.mean():.6f} (target: 0.0)")
print(f"  Std:  {train_scaled.std():.6f} (target: 1.0)")

# ── SAVE processed data ────────────────────────────────
np.savez_compressed(
    'outputs/processed_data.npz',
    train=train_scaled,
    val=val_scaled,
    test=test_scaled
)

# Save scaler
with open('outputs/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✓ Saved to outputs/:")
print(f"  processed_data.npz — {train_scaled.shape}, {val_scaled.shape}, {test_scaled.shape}")
print(f"  scaler.pkl — StandardScaler fitted on {train_scaled.shape[0]} rows")
print(f"  feature_names.json — {len(feature_cols)} features")
```

#### Cell 15: Verify Processed Data

```python
# ────────────────────────────────────────────────────────
# CELL 15: VERIFY PROCESSED DATA
# ────────────────────────────────────────────────────────
# Load lại và kiểm tra

import json, pickle

# Load
data = np.load('outputs/processed_data.npz')
train = data['train']
val = data['val']
test = data['test']

with open('outputs/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('outputs/feature_names.json') as f:
    feature_names = json.load(f)

# Verify
print("✓ Loaded processed data:")
print(f"  Train:   {train.shape}")
print(f"  Val:     {val.shape}")
print(f"  Test:    {test.shape}")
print(f"  Features: {len(feature_names)}")
print(f"\n  Train mean: {train.mean():.6f} | std: {train.std():.6f}")
print(f"  Val mean:   {val.mean():.6f} | std: {val.std():.6f}")
print(f"  Test mean:  {test.mean():.6f} | std: {test.std():.6f}")

# Inverse transform test
sample = scaler.inverse_transform(test[:5])
print(f"\n✓ Inverse transform works — sample shape: {sample.shape}")
```

---

### 4.3 `src/config.py` — Trái tim của project

```python
"""
Toàn bộ hyperparameters và cấu hình cho project.
Chỉ cần sửa file này để thay đổi thông số — không cần sửa code logic.
"""

import os

# ── Đường dẫn ──────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: processed data từ eda.ipynb
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data.npz")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, "feature_names.json")

# Output: checkpoints & results
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# ── DLinear Architecture ───────────────────────────
MODEL_TYPE = "DLinear-S"       # "DLinear-S" (shared) hoặc "DLinear-I" (individual)
KERNEL_SIZE = 25               # Moving average kernel size
                                #  7: short-term patterns (~daily)
                                #  24: matches 24h day cycle
                                #  25: paper default — good general
                                #  168: weekly patterns

# ── Time Series Windows ────────────────────────────
LOOKBACK = 336                 # Input: 14 ngày (336 giờ)
FORECAST = 96                  # Output: 4 ngày (96 giờ)

# ── Training Hyperparameters ───────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10                  # Early stopping patience
WEIGHT_DECAY = 1e-5            # L2 regularization (nhẹ, tránh overfit)

# ── Device ─────────────────────────────────────────
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# ── Random seed (reproducibility) ──────────────────
SEED = 42
```

### 4.4 `src/data_loader.py` — Data Pipeline (LOAD từ outputs/)

> **★ Nguyên tắc:** File này CHỈ load dữ liệu đã xử lý từ `outputs/` (do `eda.ipynb` export ra).
> KHÔNG đọc CSV gốc, KHÔNG tự ý scale/split — tất cả việc đó đã làm trong notebook.

```python
"""
Data Loader cho DLinear.
LOAD processed data từ outputs/ → tạo sliding windows → PyTorch DataLoader.

QUAN TRỌNG: File này không xử lý data từ đầu. Data đã được xử lý trong eda.ipynb.
Chỉ LOAD những gì notebook đã export ra outputs/.
"""

import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


class WeatherDataset(Dataset):
    """
    PyTorch Dataset: tạo sliding windows từ dữ liệu đã scale.
    
    Mỗi sample:
      X: (lookback, n_features) — cửa sổ quá khứ
      Y: (forecast, n_features) — cửa sổ tương lai cần dự đoán
    """
    
    def __init__(self, data, lookback, forecast):
        """
        Args:
            data: np.ndarray shape (T, C) — dữ liệu đã scale từ outputs/
            lookback: int
            forecast: int
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.forecast = forecast
        
    def __len__(self):
        return len(self.data) - self.lookback - self.forecast + 1
    
    def __getitem__(self, idx):
        X = self.data[idx : idx + self.lookback]
        Y = self.data[idx + self.lookback : idx + self.lookback + self.forecast]
        return X, Y


def load_processed_data():
    """
    Full pipeline: LOAD processed data từ outputs/ → DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, scaler, feature_names
    """
    print("=" * 70)
    print(" LOADING PROCESSED DATA (from eda.ipynb exports)")
    print("=" * 70)
    
    # ── BƯỚC 1: Load processed data ──────────────────
    print("[1/4] Loading processed data from outputs/...")
    data = np.load(config.PROCESSED_DATA_PATH)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    # ── BƯỚC 2: Load scaler ───────────────────────────
    print("[2/4] Loading scaler...")
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # ── BƯỚC 3: Load feature names ────────────────────
    print("[3/4] Loading feature names...")
    with open(config.FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    
    n_features = len(feature_names)
    print(f"    → Features: {n_features}")
    print(f"    → Train: {train_data.shape[0]:,} rows")
    print(f"    → Val:   {val_data.shape[0]:,} rows")
    print(f"    → Test:  {test_data.shape[0]:,} rows")
    
    # ── BƯỚC 4: Sliding windows → DataLoaders ─────────
    print("[4/4] Creating sliding windows...")
    train_dataset = WeatherDataset(train_data, config.LOOKBACK, config.FORECAST)
    val_dataset = WeatherDataset(val_data, config.LOOKBACK, config.FORECAST)
    test_dataset = WeatherDataset(test_data, config.LOOKBACK, config.FORECAST)
    
    print(f"    Train windows: {len(train_dataset):,}")
    print(f"    Val windows:   {len(val_dataset):,}")
    print(f"    Test windows:  {len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # ── Verify shape ──────────────────────────────────
    X_sample, Y_sample = next(iter(train_loader))
    print(f"\n{'─'*50}")
    print(f"Shape verification:")
    print(f"  X: {tuple(X_sample.shape)}")
    print(f"     Expected: ({config.BATCH_SIZE}, {config.LOOKBACK}, {n_features})")
    print(f"  Y: {tuple(Y_sample.shape)}")
    print(f"     Expected: ({config.BATCH_SIZE}, {config.FORECAST}, {n_features})")
    
    # Assert shapes
    assert X_sample.shape == (config.BATCH_SIZE, config.LOOKBACK, n_features), \
        f"X shape mismatch: {X_sample.shape}"
    assert Y_sample.shape == (config.BATCH_SIZE, config.FORECAST, n_features), \
        f"Y shape mismatch: {Y_sample.shape}"
    print("  ✓ All shapes correct!")
    
    return train_loader, val_loader, test_loader, scaler, feature_names


if __name__ == "__main__":
    load_processed_data()
```

### 4.5 `src/models/dlinear.py` — DLinear Model

```python
"""
DLinear: Decomposition-based Linear Model cho Time Series Forecasting.

Paper: "Are Transformers Effective for Time Series Forecasting?"
       Zeng et al., AAAI 2023

Kiến trúc:
  Input X (batch, L, C)
      │
      ├─ AvgPool1d(kernel_size) ──→ Trend ──→ Linear(L→H) ──→ Ŷ_trend
      │
      └─ X - Trend ──→ Seasonal ──→ Linear(L→H) ──→ Ŷ_seasonal
      
  Output = Ŷ_trend + Ŷ_seasonal  →  (batch, H, C)

Hai biến thể:
  - DLinear-S: Shared weights across channels (1 Linear cho tất cả)
  - DLinear-I: Individual weights per channel (C Linear layers riêng biệt)
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Trích xuất trend bằng Moving Average.
    
    Sử dụng AvgPool1d với padding = kernel_size // 2 để giữ nguyên sequence length:
      - Mỗi vị trí t là trung bình của (t-pad) đến (t+pad)
      - Tại biên: padding bù thêm giá trị 0 (giảm bias ở rìa)
    """
    
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # AvgPool1d chia cho kernel_size tự động
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, C, L) — đã transpose từ (batch, L, C)
        Returns:
            trend: (batch, C, L) — same shape
        """
        return self.avg(x)


class SeriesDecomp(nn.Module):
    """
    Phân rã chuỗi thời gian thành Trend + Seasonal.
    
    Trend = MovingAverage(X)
    Seasonal = X - Trend
    """
    
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, C, L)
        Returns:
            trend: (batch, C, L)
            seasonal: (batch, C, L)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class DLinear(nn.Module):
    """
    DLinear Model.
    
    Args:
        lookback (int):   Độ dài cửa sổ đầu vào L
        forecast (int):   Độ dài cửa sổ dự báo H
        n_features (int): Số lượng channels C
        kernel_size (int): Kernel size cho moving average (default: 25)
        individual (bool): DLinear-I (True) hay DLinear-S (False)
    """
    
    def __init__(self, lookback, forecast, n_features, kernel_size=25, individual=False):
        super().__init__()
        
        self.lookback = lookback
        self.forecast = forecast
        self.n_features = n_features
        self.individual = individual
        
        # Decomposition module
        self.decomp = SeriesDecomp(kernel_size)
        
        if individual:
            # ── DLinear-I: Mỗi channel có 2 Linear riêng ──
            # Tổng params: 2 × C × L × H
            self.linear_trend = nn.ModuleList([
                nn.Linear(lookback, forecast) for _ in range(n_features)
            ])
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(lookback, forecast) for _ in range(n_features)
            ])
        else:
            # ── DLinear-S: 1 cặp Linear cho tất cả channels ──
            # Tổng params: 2 × L × H
            self.linear_trend = nn.Linear(lookback, forecast)
            self.linear_seasonal = nn.Linear(lookback, forecast)
    
    def forward(self, x):
        """
        Args:
            x: (batch, L, C) — raw input (B, lookback, n_features)
        Returns:
            y: (batch, H, C) — forecast (B, forecast, n_features)
        """
        # ── Step 1: (B, L, C) → (B, C, L) cho Conv1d ──
        x = x.transpose(1, 2)
        
        # ── Step 2: Decomposition ──
        trend, seasonal = self.decomp(x)      # both (B, C, L)
        
        # ── Step 3: Transpose → (B, C, L) để Linear map L → H ──
        # (Linear áp dụng trên dim cuối cùng)
        trend = trend.transpose(1, 2)          # (B, L, C)
        seasonal = seasonal.transpose(1, 2)    # (B, L, C)
        
        # ── Step 4: Linear layers per channel ──
        if self.individual:
            trend_out = []
            seasonal_out = []
            for i in range(self.n_features):
                # Mỗi channel: (B, L) → Linear → (B, H)
                trend_out.append(self.linear_trend[i](trend[:, :, i]))        # (B, H)
                seasonal_out.append(self.linear_seasonal[i](seasonal[:, :, i]))  # (B, H)
            
            trend_forecast = torch.stack(trend_out, dim=-1)          # (B, H, C)
            seasonal_forecast = torch.stack(seasonal_out, dim=-1)    # (B, H, C)
        else:
            # Shared: (B, L, C) → Linear áp dụng trên dim=1 (time axis)
            trend_forecast = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
            # trend.transpose(1,2) → (B, C, L) → Linear(L→H) → (B, C, H) → transpose → (B, H, C)
            
            seasonal_forecast = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        
        # ── Step 5: Combine ──
        y = trend_forecast + seasonal_forecast    # (B, H, C)
        
        return y
    
    def count_params(self):
        """Đếm số lượng tham số"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 4.6 `src/train.py` — Training Script

```python
"""
Training script cho DLinear.

Bao gồm:
  - Training loop với validation mỗi epoch
  - Early stopping (patience-based)
  - Model checkpointing (lưu model tốt nhất)
  - Loss history logging
  - Learning rate scheduling (ReduceLROnPlateau)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

import config
from data_loader import load_processed_data
from models.dlinear import DLinear


def set_seed(seed):
    """Đảm bảo reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer):
    """Train 1 epoch, return average loss."""
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
        
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion):
    """Validate 1 epoch (no gradient), return average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def plot_training_history(history, save_path):
    """
    VẼ BIỂU ĐỒ LOSS THEO EPOCHS.
    
    Biểu đồ này CỰC KỲ QUAN TRỌNG để xác định:
      1. Model có đang học không (loss giảm?)
      2. Overfitting (train loss ↓ nhưng val loss ↑)
      3. Underfitting (cả 2 loss đều cao và không giảm)
      4. Điểm dừng tối ưu (val loss minimum)
    
    Cách đọc biểu đồ:
      ┌─────────────────────────────────────────────────┐
      │  Loss                                            │
      │   │                                              │
      │   │  train ────                                  │
      │   │            ╲                                 │
      │   │  val ─ ─ ─ ─╲─────  ←─ overfitting bắt đầu  │
      │   │              ╲       tại điểm val loss tăng  │
      │   │               ╲────                          │
      │   │                                              │
      │   └──────────────────────────────→ Epochs        │
      │          ↑ BEST MODEL ← checkpoint ở đây         │
      └─────────────────────────────────────────────────┘
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # ── Biểu đồ 1: Loss curve ────────────────────────
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # Đánh dấu điểm val loss nhỏ nhất
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = history['val_loss'][best_epoch - 1]
    ax1.scatter(best_epoch, best_val, c='red', s=100, zorder=10,
                edgecolors='darkred', linewidths=2,
                label=f'Best: Epoch {best_epoch}\nVal Loss = {best_val:.6f}')
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontsize=13, fontweight='bold')
    ax1.set_title('DLinear Training & Validation Loss', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ── Biểu đồ 2: Log-scale view (thấy rõ hơn giai đoạn hội tụ) ──
    ax2.semilogy(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax2.semilogy(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax2.scatter(best_epoch, best_val, c='red', s=100, zorder=10,
                edgecolors='darkred', linewidths=2)
    
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE Loss (log scale)', fontsize=13, fontweight='bold')
    ax2.set_title('Training & Validation Loss (Log Scale)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Loss curve saved to: {save_path}")


def train():
    """Main training function."""
    print("=" * 70)
    print(" DLinear Weather Forecasting — TRAINING")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_TYPE} | Lookback: {config.LOOKBACK}h | Forecast: {config.FORECAST}h")
    print(f"Kernel: {config.KERNEL_SIZE} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print("=" * 70)
    
    # ── Reproducibility ──
    set_seed(config.SEED)
    
    # ── Data ──
    train_loader, val_loader, test_loader, scaler, feature_names = load_processed_data()
    n_features = len(feature_names)
    
    # ── Model ──
    individual = (config.MODEL_TYPE == "DLinear-I")
    model = DLinear(
        lookback=config.LOOKBACK,
        forecast=config.FORECAST,
        n_features=n_features,
        kernel_size=config.KERNEL_SIZE,
        individual=individual
    ).to(config.DEVICE)
    
    n_params = model.count_params()
    print(f"\n📐 Model parameters: {n_params:,}")
    print(f"   (DLinear-S: ~{2 * config.LOOKBACK * config.FORECAST:,} | DLinear-I: ~{2 * n_features * config.LOOKBACK * config.FORECAST:,})")
    
    # ── Loss & Optimizer ──
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # ── Training state ──
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # ── Training Loop ──
    print(f"\n{'='*70}")
    print(f"{'Epoch':<8}{'Train Loss':<16}{'Val Loss':<16}{'LR':<12}{'Patience':<10}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        patience_str = f"{patience_counter}/{config.PATIENCE}"
        print(f"{epoch:<8}{train_loss:<16.6f}{val_loss:<16.6f}{current_lr:<12.2e}{patience_str:<10}")
        
        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history,
                'config': {k: v for k, v in vars(config).items() if not k.startswith('__')},
                'feature_names': feature_names,
            }
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⏹ Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"🏆 Best model: Epoch {best_epoch} | Val Loss: {best_val_loss:.6f}")
    print(f"💾 Checkpoint saved to: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")
    
    # ── Plot loss curve ──
    plot_training_history(history, os.path.join(config.OUTPUT_DIR, 'loss_curve.png'))
    
    # ── Save history JSON ──
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history, scaler, feature_names, test_loader


if __name__ == "__main__":
    train()
```

### 4.7 `src/evaluate.py` — Evaluation Script

```python
"""
Evaluate DLinear model trên test set.

Tính toán:
  - MSE, MAE, RMSE, MAPE cho từng feature
  - Tổng hợp metric trung bình
  - Visualize prediction vs actual
  - Naive baseline comparison (lặp giá trị cuối)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from data_loader import load_processed_data
from models.dlinear import DLinear


# ═══════════════════════════════════════════════════════════
# METRICS — Định nghĩa toán học
# ═══════════════════════════════════════════════════════════

def mse(y_true, y_pred):
    """
    Mean Squared Error:
      MSE = (1/N) · Σ(y_true_i - y_pred_i)²
    
    Đơn vị: bình phương đơn vị gốc
    Nhạy với outliers (do bình phương sai số)
    """
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """
    Mean Absolute Error:
      MAE = (1/N) · Σ|y_true_i - y_pred_i|
    
    Đơn vị: cùng đơn vị với biến gốc
    Ít nhạy với outliers hơn MSE
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error:
      RMSE = √(MSE) = √((1/N) · Σ(y_true_i - y_pred_i)²)
    
    Đơn vị: cùng đơn vị với biến gốc
    Phạt nặng các sai số lớn
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error:
      MAPE = (1/N) · Σ|(y_true_i - y_pred_i) / (y_true_i + ε)| × 100%
    
    Lưu ý: MAPE không xác định khi y_true ≈ 0 (vd: precipitation).
    Với các biến có giá trị gần 0, có thể dùng sMAPE hoặc bỏ qua.
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def compute_all_metrics(y_true, y_pred):
    """Tính tất cả metrics."""
    return {
        'MSE': mse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def naive_baseline(test_loader):
    """
    Baseline đơn giản nhất: DỰ BÁO = LẶP LẠI GIÁ TRỊ CUỐI CÙNG.
    
    Ŷ_{t+1}, ..., Ŷ_{t+H} = X_t  (lặp lại giá trị cuối H lần)
    
    Đây là lower bound — bất kỳ model ML nào cũng phải tốt hơn baseline này.
    Nếu không → model của bạn đang không học được gì.
    """
    criterion = nn.MSELoss()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for X, Y in test_loader:
            # X: (B, L, C) → lấy giá trị cuối X[:, -1, :] → (B, C)
            last_value = X[:, -1:, :]                        # (B, 1, C)
            naive_pred = last_value.repeat(1, Y.shape[1], 1)  # (B, H, C)
            loss = criterion(naive_pred, Y)
            total_loss += loss.item() * X.size(0)
            count += X.size(0)
    
    return total_loss / count


def evaluate():
    """Main evaluation function."""
    print("=" * 70)
    print(" DLinear Weather Forecasting — EVALUATION")
    print("=" * 70)
    
    # ── Load checkpoint ──
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found! Run train.py first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    feature_names = checkpoint['feature_names']
    n_features = len(feature_names)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})")
    
    # ── Data ──
    train_loader, val_loader, test_loader, scaler, _ = load_processed_data()
    
    # ── Model ──
    individual = (config.MODEL_TYPE == "DLinear-I")
    model = DLinear(
        lookback=config.LOOKBACK,
        forecast=config.FORECAST,
        n_features=n_features,
        kernel_size=config.KERNEL_SIZE,
        individual=individual
    ).to(config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # ── Collect all predictions ──
    print("\nRunning inference on test set...")
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
            Y_pred = model(X)
            all_preds.append(Y_pred.cpu().numpy())
            all_trues.append(Y.cpu().numpy())
    
    # Concatenate all batches
    Y_pred_all = np.concatenate(all_preds, axis=0)   # (N_test, H, C)
    Y_true_all = np.concatenate(all_trues, axis=0)   # (N_test, H, C)
    
    # Flatten time steps for overall metrics
    Y_pred_flat = Y_pred_all.reshape(-1, n_features)
    Y_true_flat = Y_true_all.reshape(-1, n_features)
    
    # ── Per-feature metrics ──
    print(f"\n{'='*90}")
    print(f"{'Feature':<35}{'MSE':<12}{'MAE':<12}{'RMSE':<12}{'MAPE':<12}")
    print(f"{'='*90}")
    
    results = {}
    for i, name in enumerate(feature_names):
        metrics = compute_all_metrics(Y_true_flat[:, i], Y_pred_flat[:, i])
        results[name] = metrics
        print(f"{name:<35}{metrics['MSE']:<12.4f}{metrics['MAE']:<12.4f}{metrics['RMSE']:<12.4f}{metrics['MAPE']:<12.2f}")
    
    # ── Overall average ──
    avg_mse = np.mean([r['MSE'] for r in results.values()])
    avg_mae = np.mean([r['MAE'] for r in results.values()])
    avg_rmse = np.mean([r['RMSE'] for r in results.values()])
    print(f"{'='*90}")
    print(f"{'AVERAGE':<35}{avg_mse:<12.4f}{avg_mae:<12.4f}{avg_rmse:<12.4f}{'—':<12}")
    
    # ── Baseline comparison ──
    print(f"\n{'─'*70}")
    print(" BASELINE COMPARISON")
    print(f"{'─'*70}")
    
    baseline_mse = naive_baseline(test_loader)
    print(f"Naive baseline MSE (repeat last value): {baseline_mse:.6f}")
    print(f"DLinear test MSE (overall):              {avg_mse:.6f}")
    improvement = (baseline_mse - avg_mse) / baseline_mse * 100
    print(f"Improvement:                               {improvement:.1f}%")
    
    if improvement > 0:
        print("✅ DLinear BEATS naive baseline!")
    else:
        print("❌ DLinear is WORSE than naive baseline — check your model!")
    
    # ── Save results ──
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'test_metrics.csv'))
    print(f"\n💾 Metrics saved to: {os.path.join(config.OUTPUT_DIR, 'test_metrics.csv')}")
    
    return Y_pred_all, Y_true_all, results, feature_names, scaler


if __name__ == "__main__":
    evaluate()
```

### 4.8 `src/utils.py` — Visualization & Helpers

```python
"""
Utility functions: visualization, metrics, model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch

import config


def plot_prediction_vs_actual(Y_true, Y_pred, feature_idx, feature_name, 
                                scaler, window_idx=0, save_path=None):
    """
    VẼ BIỂU ĐỒ SO SÁNH PREDICTED VS ACTUAL cho 1 feature trong 1 forecast window.
    
    Biểu đồ này trả lời câu hỏi: "Model dự đoán chính xác đến đâu trong 96h tới?"
    
    Args:
        Y_true: np.ndarray (N, H, C) — all test predictions
        Y_pred: np.ndarray (N, H, C)
        feature_idx: int — index của feature cần vẽ
        feature_name: str — tên feature
        scaler: StandardScaler — để inverse transform về đơn vị gốc
        window_idx: int — sample thứ mấy trong test set
    """
    # Inverse transform về đơn vị gốc
    # Tạo array full C columns, chỉ fill 1 cột cần vẽ
    true_full = np.zeros((Y_true.shape[1], Y_true.shape[2]))
    pred_full = np.zeros_like(true_full)
    true_full[:, feature_idx] = Y_true[window_idx, :, feature_idx]
    pred_full[:, feature_idx] = Y_pred[window_idx, :, feature_idx]
    
    true_original = scaler.inverse_transform(true_full)[:, feature_idx]
    pred_original = scaler.inverse_transform(pred_full)[:, feature_idx]
    
    hours = np.arange(1, len(true_original) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # ── Subplot 1: Actual vs Predicted ──
    ax1.plot(hours, true_original, 'b-', linewidth=2.5, label='Actual', marker='o', markersize=4)
    ax1.plot(hours, pred_original, 'r--', linewidth=2.5, label='DLinear Predicted', marker='s', markersize=4)
    ax1.fill_between(hours, true_original, pred_original, alpha=0.15, color='gray')
    ax1.set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold')
    ax1.set_ylabel(feature_name, fontsize=13, fontweight='bold')
    ax1.set_title(f'DLinear Forecast: {feature_name} (96h ahead)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # ── Subplot 2: Error over time ──
    error = true_original - pred_original
    colors = ['red' if e < 0 else 'blue' for e in error]
    ax2.bar(hours, error, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax2.set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'Error ({feature_name})', fontsize=13, fontweight='bold')
    ax2.set_title(f'Prediction Error Over Forecast Horizon', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_learned_weights(model, feature_names, save_path=None):
    """
    VẼ TRỌNG SỐ ĐÃ HỌC CỦA DLINEAR.
    
    Trọng số W ∈ R^{H×L} cho biết mỗi bước trong quá khứ (L)
    đóng góp bao nhiêu vào mỗi bước dự báo (H).
    
    Cách đọc heatmap:
      - Trục X: vị trí trong quá khứ (1 = 336h trước, 336 = 1h trước)
      - Trục Y: bước dự báo (1 = 1h tới, 96 = 96h tới)
      - Màu đỏ (dương): giá trị quá khứ cao → dự báo cao
      - Màu xanh (âm): giá trị quá khứ cao → dự báo thấp
      - Màu trắng (≈0): không ảnh hưởng
    """
    if hasattr(model, 'linear_trend'):
        W_trend = model.linear_trend.weight.detach().cpu().numpy()  # (H, L)
        W_seasonal = model.linear_seasonal.weight.detach().cpu().numpy()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Trend weights
        im1 = ax1.imshow(W_trend, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
        ax1.set_title('Trend Branch Weights\n($W_t$: L → H mapping)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Past timestep (L=336 → 0h ago)')
        ax1.set_ylabel('Forecast step (H=1 → 96h ahead)')
        plt.colorbar(im1, ax=ax1)
        
        # Seasonal weights
        im2 = ax2.imshow(W_seasonal, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
        ax2.set_title('Seasonal Branch Weights\n($W_s$: L → H mapping)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Past timestep (L=336 → 0h ago)')
        ax2.set_ylabel('Forecast step (H=1 → 96h ahead)')
        plt.colorbar(im2, ax=ax2)
        
        # Difference (seasonal - trend)
        W_diff = W_seasonal - W_trend
        im3 = ax3.imshow(W_diff, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
        ax3.set_title('Weight Difference\n($W_s - W_t$: what seasonal adds)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Past timestep (L=336 → 0h ago)')
        ax3.set_ylabel('Forecast step (H=1 → 96h ahead)')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def plot_metrics_bar_chart(results, save_path=None):
    """
    VẼ BIỂU ĐỒ CỘT SO SÁNH METRICS GIỮA CÁC FEATURES.
    
    Giúp trả lời: "Biến nào khó dự đoán nhất? Biến nào dễ nhất?"
    """
    features = list(results.keys())
    mae_values = [results[f]['MAE'] for f in features]
    
    # Sort by MAE (descending)
    sorted_idx = np.argsort(mae_values)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    mae_sorted = [mae_values[i] for i in sorted_idx]
    
    # Color gradient: dễ (xanh) → khó (đỏ)
    norm = plt.Normalize(min(mae_sorted), max(mae_sorted))
    colors = plt.cm.RdYlGn_r(norm(mae_sorted))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(features_sorted, mae_sorted, color=colors, edgecolor='black', linewidth=0.5)
    
    # Ghi giá trị lên thanh
    for bar, val in zip(bars, mae_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('MAE (Mean Absolute Error — lower is better)', fontsize=13, fontweight='bold')
    ax.set_title('DLinear Performance by Feature (MAE on Test Set)', fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 4.9 `README.md`

```markdown
# Weather Forecasting with DLinear

Dự báo thời tiết đa biến sử dụng DLinear — mô hình tuyến tính phân rã 
(Decomposition Linear) cho Long-term Time Series Forecasting.

Paper: [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504)
(Zeng et al., AAAI 2023)

## Quick Start

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy data pipeline test
python src/data_loader.py

# 3. Training
python src/train.py

# 4. Evaluation
python src/evaluate.py
```

## Project Structure

Xem `plan/dlinear_implementation_plan.md` để biết chi tiết kiến trúc, 
toán học và thuật toán.
```

---

## 5. Metric đánh giá

### 5.1 Công thức toán học

| Metric | Công thức | Range | Interpretation |
|--------|-----------|-------|----------------|
| **MSE** | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | $[0, \infty)$ | Bình phương sai số; nhạy với outliers |
| **MAE** | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$ | $[0, \infty)$ | Trung bình độ lớn sai số; cùng đơn vị với biến |
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ | $[0, \infty)$ | Căn MSE; phạt nặng sai số lớn; cùng đơn vị với biến |
| **MAPE** | $\frac{1}{N}\sum\|\frac{y_i - \hat{y}_i}{y_i}\| \times 100\%$ | $[0, \infty)$ | Sai số %; không dùng được khi y ≈ 0 |

### 5.2 Cách diễn giải kết quả

| Tình huống | Dấu hiệu | Hành động |
|------------|----------|-----------|
| **Model tốt** | DLinear MSE << Naive MSE, loss curve hội tụ mượt | ✅ Triển khai |
| **Overfitting** | Train loss ↓↓, Val loss ↑↑ sau 1 điểm | Tăng weight_decay, giảm model size, thêm dropout |
| **Underfitting** | Cả 2 loss cao, không giảm | Tăng lookback, tăng epochs, thử DLinear-I |
| **Không học** | Loss phẳng, ~= baseline | Kiểm tra data pipeline, normalization, learning rate |
| **Diverging** | Loss → ∞ | Giảm LR, kiểm tra gradient explosion |

### 5.3 Baseline để so sánh

Luôn so sánh với ít nhất 2 baseline:

1. **Naive (Repeat Last):** $\hat{Y}_{t+1},...,\hat{Y}_{t+H} = X_t$ — lặp giá trị cuối
2. **Linear Regression:** $\hat{Y} = W \cdot X + b$ (không decomposition) — để kiểm tra decomposition có thực sự giúp ích

---

## 6. Biểu đồ & Trực quan hóa

### 6.1 Loss Curve (BIỂU ĐỒ QUAN TRỌNG NHẤT)

```
Loss ↑
     │
     │  ╲
     │   ╲___           ←─ Train loss: giảm đều, hội tụ
     │       ╲___
     │  ─ ─ ─ ─╲────    ←─ Val loss: đi theo train, có thể nhiễu hơn
     │          ╲___
     │             ╲___ ←─ Điểm optimal: val loss nhỏ nhất
     │                 ╲___ ←─ Overfitting: val loss tăng nhẹ
     │
     └────────────────────────────→ Epochs
              ↑
         BEST MODEL — checkpoint ở đây
```

**Cách dùng:**
- Nếu val loss đã plateau (đi ngang) 10+ epochs → có thể dừng sớm
- Khoảng cách train-val càng lớn → overfitting càng nặng
- Nếu loss curve rung lắc mạnh → giảm learning rate hoặc tăng batch size

### 6.2 Weight Visualization

Heatmap $W \in \mathbb{R}^{H \times L}$ cho thấy mô hình "nhìn" vào đâu trong quá khứ:

```
Forecast step (H=96)
  ↑
  │  ┌────────────────────────────┐
  │  │ ░░░░████████░░░░░░░░░░░░░░ │ ←─ Gần tương lai: dùng chủ yếu
  │  │ ░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░ │    thông tin gần đây (phải)
  │  │ ░░░░░░░░▒▒▒▒▒▒░░░░░░░░░░░ │
  │  │ ░░░░░░░░░░░░▒▒▒▒▒▒░░░░░░░ │ ←─ Xa tương lai: dùng thêm
  │  │ ░░░░░░░░░░░░░░░░▒▒▒▒░░░░░ │    thông tin xa hơn (trái)
  │  └────────────────────────────┘
  └──────────────────────────────────→ Past step (L=336)
     xa quá khứ               gần đây
```

**Cách đọc:**
- Màu đỏ đậm ở cột bên PHẢI → model dựa nhiều vào thông tin gần đây (hợp lý)
- Pattern lặp mỗi 24 cột → model học được chu kỳ ngày
- Pattern lặp mỗi 168 cột → model học được chu kỳ tuần

### 6.3 Actual vs Predicted

Biểu đồ đường so sánh giá trị thực tế và dự báo trên toàn bộ 96h forecast horizon. Cần kiểm tra:
- Model có bắt được xu hướng không (biên độ, pha)
- Sai số tăng dần theo forecast horizon (càng xa càng khó — bình thường)
- Có systematic bias không (luôn dự đoán cao hơn/thấp hơn)

### 6.4 Feature Importance (MAE bar chart)

Bar chart ngang xếp hạng các biến theo MAE:
- Biến dễ: soil temperature, pressure (ít biến động)
- Biến khó: precipitation, cloud cover, wind gusts (biến động mạnh, nhiễu)

---

## 7. Script chạy & Độ chính xác

### 7.1 Quy trình chạy đầy đủ

> **★ Luồng làm việc:** `eda.ipynb` (data processing) → `data_loader.py` (load) → `train.py` → `evaluate.py`

```powershell
# ═══════════════════════════════════════════════════════
# BƯỚC 0: Cài đặt môi trường
# ═══════════════════════════════════════════════════════
cd e:\WeatherForecast
pip install -r requirements.txt

# ═══════════════════════════════════════════════════════
# BƯỚC 1: Chạy eda.ipynb (Data Processing)
# ═══════════════════════════════════════════════════════
# Mở Jupyter: jupyter notebook
# Chạy eda.ipynb từ đầu đến cuối (Run All Cells)
# → Cells 0-11: EDA (phân tích, biểu đồ)
# → Cells 12-15: Data Processing MỚI THÊM VÀO
#     Cell 12: Feature Selection
#     Cell 13: Feature Engineering (cyclic encoding)
#     Cell 14: Split + Scale → Export
#     Cell 15: Verify
#
# Expected output files in outputs/:
#   ✓ processed_data.npz    (~30MB)
#   ✓ scaler.pkl            (~2KB)
#   ✓ feature_names.json    (~1KB)

# ═══════════════════════════════════════════════════════
# BƯỚC 2: Test data loader (đảm bảo load đúng)
# ═══════════════════════════════════════════════════════
python src/data_loader.py

# Expected output:
# ======================================================================
#  LOADING PROCESSED DATA (from eda.ipynb exports)
# ======================================================================
# [1/4] Loading processed data from outputs/...
# [2/4] Loading scaler...
# [3/4] Loading feature names...
#     → Features: 22
#     → Train: 30,677 rows
#     → Val:   6,574 rows
#     → Test:  6,573 rows
# [4/4] Creating sliding windows...
#     Train windows: 30,245
#     Val windows:   6,142
#     Test windows:  6,141
# --------------------------------------------------
# Shape verification:
#   X: (64, 336, 22)
#      Expected: (64, 336, 22)
#   Y: (64, 96, 22)
#      Expected: (64, 96, 22)
#   ✓ All shapes correct!

# ═══════════════════════════════════════════════════════
# BƯỚC 3: Training
# ═══════════════════════════════════════════════════════
python src/train.py

# Expected output:
# ======================================================================
#  DLinear Weather Forecasting — TRAINING
# ======================================================================
# Device: cpu
# Model: DLinear-S | Lookback: 336h | Forecast: 96h
# Kernel: 25 | Batch: 64 | LR: 0.001
# ======================================================================
# ... (data loading logs) ...
# 
# 📐 Model parameters: 64,512
#    (DLinear-S: ~64,512 | DLinear-I: ~1,290,240)
# 
# ======================================================================
# Epoch   Train Loss      Val Loss        LR          Patience  
# ======================================================================
# 1       0.852416        0.723145        1.00e-03    0/10
# 2       0.561234        0.489012        1.00e-03    0/10
# ...
# 25      0.084512        0.092145        1.00e-04    0/10
# 26      0.083901        0.092356        1.00e-04    1/10
# ...
# 35      0.078234        0.093456        5.00e-05    10/10
# 
# ⏹ Early stopping triggered at epoch 35
# 
# ======================================================================
# ✅ Training completed in 245.3s (4.1 min)
# 🏆 Best model: Epoch 25 | Val Loss: 0.092145
# 💾 Checkpoint saved to: checkpoints/best_model.pth
# 
# 📈 Loss curve saved to: outputs/loss_curve.png

# ═══════════════════════════════════════════════════════
# BƯỚC 4: Evaluation
# ═══════════════════════════════════════════════════════
python src/evaluate.py

# Expected output:
# ======================================================================
#  DLinear Weather Forecasting — EVALUATION
# ======================================================================
# Loaded model from epoch 25 (val_loss=0.092145)
# ...
# Running inference on test set...
# 
# ==========================================================================================
# Feature                            MSE         MAE         RMSE        MAPE        
# ==========================================================================================
# temperature_2m (°C)                0.0521      0.1602      0.2283      0.95        
# relative_humidity_2m (%)           0.0891      0.2123      0.2985      1.34        
# dew_point_2m (°C)                  0.0612      0.1756      0.2474      1.12        
# ... (22 features)
# ==========================================================================================
# AVERAGE                            0.0712      0.1890      0.2668      —           
# 
# ──────────────────────────────────────────────────────────────────
#  BASELINE COMPARISON
# ──────────────────────────────────────────────────────────────────
# Naive baseline MSE (repeat last value): 0.124500
# DLinear test MSE (overall):              0.071200
# Improvement:                               42.8%
# ✅ DLinear BEATS naive baseline!
```

### 7.2 Kết quả kỳ vọng

Dựa trên paper gốc và các implementation tham khảo:

| Chỉ số | Kỳ vọng | Ghi chú |
|--------|---------|---------|
| Train time | 5-15 phút (CPU) / 1-3 phút (GPU) | Với EPOCHS=50, có thể early stop sớm hơn |
| Val loss hội tụ | ~epoch 20-30 | Với PATIENCE=10 |
| MSE cải thiện vs naive | 20-50% | Tùy thuộc vào feature |
| DLinear beats Linear | 5-15% | Decomposition giúp cải thiện |
| Overfitting gap | <10% chênh lệch train-val | Nếu >20% → cân nhắc regularization |

### 7.3 Các thí nghiệm ablation nên chạy

Sau khi có baseline DLinear-S, chạy thêm các biến thể để hiểu rõ model:

| Thí nghiệm | Thay đổi | Mục đích |
|------------|----------|----------|
| **Kernel size** | k=7, 24, 25, 168 | Tìm kernel tối ưu cho weather data |
| **Lookback** | L=168, 336, 720 | Input dài hơn có giúp không? |
| **DLinear-I** | individual=True | Features khác biệt có cần weights riêng? |
| **NLinear** | Thêm normalization | Có giúp với distribution shift? |
| **Linear only** | Bỏ decomposition | Decomposition có thực sự cần? |

---

## 8. Verification Checklist

Trước khi coi model là "xong", kiểm tra từng mục:

- [ ] **Data Pipeline Shape:** X = (B, L, C), Y = (B, H, C) — không bị lệch
- [ ] **No Data Leakage:** Train/val/test không overlap về mặt thời gian
- [ ] **Scaling Correct:** Train mean ≈ 0, std ≈ 1; val/test dùng scaler từ train
- [ ] **Model Forward Pass:** Output shape (B, H, C) khớp với target
- [ ] **Overfit Test:** Train trên 1 batch nhỏ → loss → 0 (chứng minh model đủ capacity)
- [ ] **Loss Convergence:** Train & val loss giảm và hội tụ (không NaN, không ∞)
- [ ] **DLinear > Naive:** MSE phải thấp hơn baseline (nếu không → bug)
- [ ] **DLinear > Linear:** Decomposition giúp cải thiện ít nhất 5%
- [ ] **Prediction Visualization:** Biểu đồ actual vs predicted trông hợp lý
- [ ] **No Systematic Bias:** Trung bình error ≈ 0 (không luôn dự đoán cao/thấp hơn)
- [ ] **Reproducibility:** Chạy 2 lần với cùng seed → kết quả giống nhau
