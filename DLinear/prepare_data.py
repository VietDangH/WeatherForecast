"""
Data Preparation — theo Weather Forecasting Shared Protocol (§2-4).

Chạy file này trước khi train:
    cd DLinear && python prepare_data.py

Output (trong DLinear/outputs/):
    processed_data.npz   — train/val/test arrays (scaled)
    scaler.pkl           — StandardScaler fitted on train
    feature_names.json   — 12 input feature names
    target_names.json    — 7 target names
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import StandardScaler

import config

# ===============================================================
# §2. Preprocessing (identical for all models)
# ===============================================================

print("=" * 70)
print(" DATA PREPARATION — Weather Forecasting Protocol")
print("=" * 70)

# 1. Load, parse time, sort
print("[1/6] Loading & sorting...")
df = pd.read_csv(config.DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df = df.set_index('time')
# Enforce hourly frequency (already hourly per protocol)
df = df.asfreq('h')
print(f"    {len(df):,} rows, {df.index[0]} -> {df.index[-1]}")

# 2. Sensor-error guard: clip wind columns >= 0
print("[2/6] Clipping wind columns >= 0...")
wind_cols = ['wind_speed_10m (km/h)', 'wind_gusts_10m (km/h)']
for c in wind_cols:
    if c in df.columns:
        before = (df[c] < 0).sum()
        df[c] = df[c].clip(lower=0)
        print(f"    {c}: {before} negative values clipped")

# 3. Wind decomposition -> Cartesian (wx, wy)
print("[3/6] Wind decomposition -> wx, wy...")
wind_speed = df['wind_speed_10m (km/h)'].values
wind_dir = np.deg2rad(df['wind_direction_10m (°)'].values)
df['wx'] = wind_speed * np.cos(wind_dir)
df['wy'] = wind_speed * np.sin(wind_dir)
print(f"    wx, wy created from wind_speed_10m x wind_direction_10m")

# 4. Time-interpolate gaps + forward/back fill
print("[4/6] Interpolating gaps...")
nulls_before = df.isnull().sum().sum()
if nulls_before > 0:
    df = df.interpolate(method='time').ffill().bfill()
    nulls_after = df.isnull().sum().sum()
    print(f"    {nulls_before} -> {nulls_after} nulls after interpolation")
else:
    print(f"    No gaps found — clean dataset")

# 5. Cyclical time encodings (ML/DL only)
print("[5/6] Adding cyclical time encodings...")
df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)
df['sin_doy'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df['cos_doy'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
print(f"    sin_hour, cos_hour, sin_doy, cos_doy added")

# ===============================================================
# §3. Data split (chronological by year — NEVER shuffle)
# ===============================================================

print(f"\n[6/6] Chronological split 60/20/20 by year...")

train_years = config.PROTOCOL_TRAIN_YEARS
train_mask = df.index.year.isin(train_years)
val_mask   = df.index.year == config.PROTOCOL_VAL_YEAR
test_mask  = df.index.year == config.PROTOCOL_TEST_YEAR

train_df = df[train_mask]
val_df   = df[val_mask]
test_df  = df[test_mask]

print(f"    Train: {len(train_df):,} rows ({train_years})")
print(f"    Val:   {len(val_df):,} rows ({config.PROTOCOL_VAL_YEAR})")
print(f"    Test:  {len(test_df):,} rows ({config.PROTOCOL_TEST_YEAR})")

# ===============================================================
# Extract input (12) & target (7) arrays
# ===============================================================

input_cols = config.INPUT_FEATURES    # 12 features
target_cols = config.TARGET_FEATURES   # 7 targets

X_train_raw = train_df[input_cols].values.astype(np.float32)
X_val_raw   = val_df[input_cols].values.astype(np.float32)
X_test_raw  = test_df[input_cols].values.astype(np.float32)

Y_train_raw = train_df[target_cols].values.astype(np.float32)
Y_val_raw   = val_df[target_cols].values.astype(np.float32)
Y_test_raw  = test_df[target_cols].values.astype(np.float32)

print(f"\n    Input shape:  X_train {X_train_raw.shape}, X_val {X_val_raw.shape}, X_test {X_test_raw.shape}")
print(f"    Target shape: Y_train {Y_train_raw.shape}, Y_val {Y_val_raw.shape}, Y_test {Y_test_raw.shape}")

# ===============================================================
# §4. Normalization (Scaler on train only)
# ===============================================================

# Scale ALL features (input + target) together for consistency
# Fit on train inputs+targets, then apply per-split
all_train = np.hstack([X_train_raw, Y_train_raw])
all_val   = np.hstack([X_val_raw, Y_val_raw])
all_test  = np.hstack([X_test_raw, Y_test_raw])

scaler = StandardScaler()
all_train_scaled = scaler.fit_transform(all_train)
all_val_scaled   = scaler.transform(all_val)
all_test_scaled  = scaler.transform(all_test)

print(f"\n    Scaler fitted on train: mean={all_train_scaled.mean():.6f}, std={all_train_scaled.std():.6f}")

# Split back into X and Y
n_input = len(input_cols)
X_train = all_train_scaled[:, :n_input]
X_val   = all_val_scaled[:, :n_input]
X_test  = all_test_scaled[:, :n_input]

Y_train = all_train_scaled[:, n_input:]
Y_val   = all_val_scaled[:, n_input:]
Y_test  = all_test_scaled[:, n_input:]

# ===============================================================
# Export
# ===============================================================

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

np.savez_compressed(config.PROCESSED_DATA_PATH,
    X_train=X_train, X_val=X_val, X_test=X_test,
    Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)

with open(config.SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

with open(config.FEATURE_NAMES_PATH, 'w') as f:
    json.dump(input_cols, f, indent=2)

with open(config.TARGET_NAMES_PATH, 'w') as f:
    json.dump(target_cols, f, indent=2)

print(f"\n{'='*70}")
print(f" [OK] Exported to {config.OUTPUT_DIR}/")
print(f"   processed_data.npz")
print(f"   scaler.pkl")
print(f"   feature_names.json ({len(input_cols)} inputs)")
print(f"   target_names.json ({len(target_cols)} targets)")
print(f"\n Input:  {X_train.shape[1]} features x {config.LOOKBACK}h lookback")
print(f" Target: {Y_train.shape[1]} variables x {config.FORECAST}h forecast")
print(f" Ready for DLinear training!")
