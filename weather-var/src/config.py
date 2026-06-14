"""Central configuration for the VAR weather-forecasting pipeline.

Dataset note: the spec was written against Jena Climate, but this repo ships an
Open-Meteo hourly dataset. We keep the spec's VAR methodology and adapt the
column mapping below to the actual data.
"""
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SRC_DIR)            # weather-var/
REPO_ROOT = os.path.dirname(PKG_ROOT)          # d:\WeatherForecast
DATA_PATH = os.path.join(REPO_ROOT, "weather_dataset.csv")
RESULTS_DIR = os.path.join(PKG_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# ----------------------------------------------------------------------------
# Chronological split (NEVER shuffle)
# ----------------------------------------------------------------------------
TRAIN_YEARS = [2021, 2022, 2023]   # VAR is fit here
VAL_YEARS = [2024]                 # reserved for group parity; VAR does not use it
TEST_YEARS = [2025]                # untouched until final rolling eval

# ----------------------------------------------------------------------------
# Forecast / model
# ----------------------------------------------------------------------------
HORIZON = 24        # forecast 24 hours ahead (shared with the group)
MAXLAGS = 48        # upper bound for AIC lag-order search
ROLL_STEP = 24      # rolling-origin step: 24 = non-overlapping, 1 = overlapping
DIFF_ORDER = 1      # default first difference; confirmed/adjusted by ADF at runtime
STANDARDIZE = False # VAR doesn't require scaling; enable for numerical stability

# ----------------------------------------------------------------------------
# Raw -> clean column names (matched on the prefix before the unit, so encoding
# of the degree/cube symbols in the CSV header is irrelevant).
# ----------------------------------------------------------------------------
PREFIX_RENAME = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "dew_point_2m": "dew",
    "precipitation": "precip",
    "surface_pressure": "pressure",
    "cloud_cover": "cloud",
    "cloud_cover_low": "cloud_low",
    "cloud_cover_mid": "cloud_mid",
    "cloud_cover_high": "cloud_high",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
    "wind_gusts_10m": "wind_gust",
    "soil_moisture_0_to_7cm": "soil",
}

# ----------------------------------------------------------------------------
# The 7 modeled weather variables.
# EDIT THIS LIST to match the group's agreed target set exactly.
# Available clean names: temp, rh, dew, precip, pressure, cloud, cloud_low,
# cloud_mid, cloud_high, wind_speed, wind_dir, wind_gust, soil, wx, wy
# ----------------------------------------------------------------------------
TARGETS = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]
