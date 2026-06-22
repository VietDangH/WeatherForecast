import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from . import config

def load_and_preprocess_data():
    file_path = config.FILE_NAME 

    df = pd.read_csv(file_path)

    print("\n1. Đang xử lý dữ liệu và tạo 11 features đầu vào...")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    df = df.astype('float32')

    df.rename(columns={
        'temperature_2m (°C)': 'temp',
        'relative_humidity_2m (%)': 'rh',
        'dew_point_2m (°C)': 'dew',
        'surface_pressure (hPa)': 'pressure',
        'cloud_cover (%)': 'cloud'
    }, inplace=True)

    df['wind_speed'] = df['wind_speed_10m (km/h)'].clip(lower=0)
    df['wind_dir'] = df['wind_direction_10m (°)']
    df['wx'] = df['wind_speed'] * np.cos(np.radians(df['wind_dir']))
    df['wy'] = df['wind_speed'] * np.sin(np.radians(df['wind_dir']))

    hour = df.index.hour
    day_of_year = df.index.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df

def split_and_scale(df):
    train_df = df.loc['2021-01-01':'2023-12-31']
    val_df   = df.loc['2024-01-01':'2024-12-31']
    test_df  = df.loc['2025-01-01':'2025-12-31']

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_df[config.FEATURES])
    train_y_scaled = scaler_y.fit_transform(train_df[config.TARGETS])
    val_X_scaled = scaler_X.transform(val_df[config.FEATURES])
    val_y_scaled = scaler_y.transform(val_df[config.TARGETS])
    test_X_scaled = scaler_X.transform(test_df[config.FEATURES])
    test_y_scaled = scaler_y.transform(test_df[config.TARGETS])

    return train_X_scaled, train_y_scaled, val_X_scaled, val_y_scaled, test_X_scaled, test_y_scaled, scaler_y

def create_windows(X, y, lookback=120, horizon=24, step=1):
    X_win, y_win = [], []
    for i in range(0, len(X) - lookback - horizon + 1, step):
        X_win.append(X[i:i+lookback])
        y_win.append(y[i+lookback:i+lookback+horizon])
    return np.array(X_win), np.array(y_win)