from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from . import config as C

# Load + Preprocess data
def load_clean():
    df = pd.read_csv(C.DATA_CSV)
    df = df.rename(columns=C.RENAME)

    # Parse time, set as index, sort ascending, enforce hourly frequency
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = df.sort_values(C.TIME_COL).set_index(C.TIME_COL)
    df = df.asfreq("h") 

    # Remove negative wind measurements
    for col in ("wind_speed", "wind_gust"):
        df[col] = df[col].clip(lower=0)

    # Wind decomposition: convert circular wind to Cartesian
    rad = np.radians(df["wind_dir"].to_numpy())
    df["wx"] = df["wind_speed"] * np.cos(rad)
    df["wy"] = df["wind_speed"] * np.sin(rad)

    # Fill any missing observations using time interpolation
    df = df.interpolate(method="time").ffill().bfill()

    # Cyclical time encodings
    hour = df.index.hour.to_numpy()
    doy = df.index.dayofyear.to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df[C.INPUT_FEATURES]

# Split + scale  
@dataclass
class Dataset:
    index: pd.DatetimeIndex      # Timestamps
    feat_scaled: np.ndarray      # (T, 11) Standardlized weather features + raw cyclic encodings
    eng_feat: np.ndarray         # (T, F) Engineered features for model
    eng_names: list              # length-F Names of engineered features
    targ_scaled: np.ndarray      # (T, 7) scaled targets
    targ_real: np.ndarray        # (T, 7) real-scale targets
    target_mean: np.ndarray      # (7,) mean for denormalisation
    target_scale: np.ndarray     # (7,) std for denormalisation
    scaler: StandardScaler

    # Inverted scale
    def denorm(self, y_scaled: np.ndarray):
        return y_scaled * self.target_scale + self.target_mean

# Engineered features
_LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 119]  
_RMEAN = [3, 6, 12, 24, 72]
_RSTD = [6, 24, 120]


def engineer_features(feat_scaled):
    df = pd.DataFrame(feat_scaled)
    cols: dict[str, pd.Series] = {}

    for c, name in enumerate(C.WEATHER_FEATURES):
        s = df[c]
        for L in _LAGS:
            cols[f"{name}_lag{L}"] = s.shift(L)
        for w in _RMEAN:
            cols[f"{name}_mean{w}"] = s.rolling(w).mean()
        for w in _RSTD:
            cols[f"{name}_std{w}"] = s.rolling(w).std()

        cols[f"{name}_min24"] = s.rolling(24).min()
        cols[f"{name}_max24"] = s.rolling(24).max()

    n_weather = len(C.WEATHER_FEATURES)

    for c, name in enumerate(C.TIME_FEATURES):
        cols[f"{name}_now"] = df[n_weather + c]

    eng = pd.DataFrame(cols)

    return eng.to_numpy(dtype=np.float32), list(eng.columns)


def build_dataset(df: pd.DataFrame) -> Dataset:
    years = df.index.year.to_numpy()
    
    # Fit scaling on TRAIN years only
    train_mask = np.isin(years, C.TRAIN_YEARS)
    scaler = StandardScaler().fit(df.loc[train_mask, C.WEATHER_FEATURES].to_numpy())

    # Scale weather variables
    weather_scaled = scaler.transform(df[C.WEATHER_FEATURES].to_numpy())

    time_enc = df[C.TIME_FEATURES].to_numpy()                
    feat_scaled = np.hstack([weather_scaled, time_enc]).astype(np.float32)

    # Extract scaling statistics for target variables only
    target_idx = [C.WEATHER_FEATURES.index(t) for t in C.TARGETS]
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx]

    targ_real = df[C.TARGETS].to_numpy().astype(np.float32)
    # Scale targets
    targ_scaled = ((targ_real - target_mean) / target_scale).astype(np.float32)

    eng_feat, eng_names = engineer_features(feat_scaled)

    return Dataset(
        index=df.index,
        feat_scaled=feat_scaled,
        eng_feat=eng_feat,
        eng_names=eng_names,
        targ_scaled=targ_scaled,
        targ_real=targ_real,
        target_mean=target_mean.astype(np.float32),
        target_scale=target_scale.astype(np.float32),
        scaler=scaler,
    )

# Origins + windowing
def get_origins(index: pd.DatetimeIndex, years, step: int, require_input_in_set: bool) -> np.ndarray:
    in_set = np.isin(index.year.to_numpy(), years)
    T = len(index)
    lb, h = C.LOOKBACK, C.HORIZON

    tgt_ok = sliding_window_view(in_set, h).all(axis=1)        # by target start
    e_range = np.arange(lb - 1, T - h)
    mask = tgt_ok[e_range + 1]
    if require_input_in_set:
        inp_ok = sliding_window_view(in_set, lb).all(axis=1)   # by input start
        mask &= inp_ok[e_range - lb + 1]

    origins_all = e_range[mask]

    return origins_all[::step]


def make_windows(ds: Dataset, origins: np.ndarray):
    h = C.HORIZON
    targ_s_win = sliding_window_view(ds.targ_scaled, (h, ds.targ_scaled.shape[1]))[:, 0]
    targ_r_win = sliding_window_view(ds.targ_real, (h, ds.targ_real.shape[1]))[:, 0]

    X = ds.eng_feat[origins]                        # window ending at origin e
    Y_scaled = targ_s_win[origins + 1]              # target starts at e+1
    Y_real = targ_r_win[origins + 1]
    return X, Y_scaled, Y_real


def split_windows(ds: Dataset):
    tr_o = get_origins(ds.index, C.TRAIN_YEARS, C.TRAIN_STEP, require_input_in_set=True)
    va_o = get_origins(ds.index, C.VAL_YEARS, C.VAL_STEP, require_input_in_set=False)
    te_o = get_origins(ds.index, C.TEST_YEARS, C.TEST_STEP, require_input_in_set=False)

    train = make_windows(ds, tr_o)
    val = make_windows(ds, va_o)
    test = make_windows(ds, te_o)
    return (tr_o, train), (va_o, val), (te_o, test)
