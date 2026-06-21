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
MAXLAGS = 60        # upper bound for AIC lag-order search (AIC settles at p=50,
                    # an interior minimum; 60 leaves headroom so it isn't a ceiling)
ROLL_STEP = 24      # rolling-origin step: 24 = non-overlapping, 1 = overlapping
DIFF_ORDER = 1      # default first difference; confirmed/adjusted by ADF at runtime
STANDARDIZE = False # VAR doesn't require scaling; enable for numerical stability

# ----------------------------------------------------------------------------
# Raw -> clean column names (matched on the prefix before the unit, so encoding
# of the degree/cube symbols in the CSV header is irrelevant).
# Only the columns this project uses are mapped: the 5 direct targets plus
# wind_speed/wind_dir (sources for the engineered wx/wy). Any other raw columns
# in the CSV are loaded but ignored.
# ----------------------------------------------------------------------------
PREFIX_RENAME = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "dew_point_2m": "dew",
    "surface_pressure": "pressure",
    "cloud_cover": "cloud",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_dir",
}

# ----------------------------------------------------------------------------
# The 7 modeled weather variables.
# Clean names: temp, rh, dew, pressure, cloud (direct) + wx, wy (from wind).
# ----------------------------------------------------------------------------
TARGETS = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]
