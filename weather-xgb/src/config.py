from __future__ import annotations
from pathlib import Path

# Paths
PKG_DIR = Path(__file__).resolve().parent          # weather-xgb/src
MODEL_DIR = PKG_DIR.parent                          # weather-xgb
ROOT = MODEL_DIR.parent                             # project root

DATA_CSV = ROOT / "weather_dataset.csv"

RESULTS_DIR = MODEL_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = MODEL_DIR / "models"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
DIAGNOSTICS_TXT = RESULTS_DIR / "diagnostics.txt"

# Rename columns
RENAME = {
    "temperature_2m (°C)": "temp",
    "relative_humidity_2m (%)": "rh",
    "dew_point_2m (°C)": "dew",
    "precipitation (mm)": "precip",
    "surface_pressure (hPa)": "pressure",
    "cloud_cover (%)": "cloud",
    "cloud_cover_low (%)": "cloud_low",
    "cloud_cover_mid (%)": "cloud_mid",
    "cloud_cover_high (%)": "cloud_high",
    "wind_speed_10m (km/h)": "wind_speed",
    "wind_direction_10m (°)": "wind_dir",
    "wind_gusts_10m (km/h)": "wind_gust",
    "soil_moisture_0_to_7cm (m³/m³)": "soil",
}

TIME_COL = "time"

# Target output is (HORIZON, len(TARGETS)) = (24, 7)
TARGETS = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]

# Input weather features
WEATHER_FEATURES = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]

# 4 cyclical time encodings
TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

# Full ordered input feature list -> 11 columns
INPUT_FEATURES = WEATHER_FEATURES + TIME_FEATURES
assert len(INPUT_FEATURES) == 11, "ML/DL input must be 11 features per the protocol"
assert all(t in WEATHER_FEATURES for t in TARGETS), "targets must be a subset of weather features"

# Windowing: input (120, 11) -> target (24, 7)
LOOKBACK = 120
HORIZON = 24

# Sliding-window step (origins per hour) for building each split.
TRAIN_STEP = 6
VAL_STEP = 24
TEST_STEP = 24

# Data splitting
TRAIN_YEARS = (2021, 2022, 2023)
VAL_YEARS = (2024,)
TEST_YEARS = (2025,)

# Model's hyper-parameters
XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_estimators=400,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=5.0,
    reg_lambda=1.0,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 25

# Seeds
SEEDS = [0, 1, 2]

# Diagnostics / evaluation knobs
LJUNGBOX_LAG = 24          # PROTOCOL.md §6
BOOTSTRAP_N = 1000         # bootstrap resamples for metric CIs (§6)
BOOTSTRAP_CI = 0.95
INTERVAL_LEVEL = 0.95      # empirical forecast-interval coverage target (§6)


def ensure_dirs():
    """Create output directories if missing."""
    for d in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
