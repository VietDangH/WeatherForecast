from __future__ import annotations

from pathlib import Path


HOURLY_FREQ = "h"

RAW_PREFIXES = {
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

TARGET_COLUMNS = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]

EXPECTED_SPLIT_ROWS = {
    "train": 26_280,
    "val": 8_784,
    "test": 8_760,
}

TRAIN_START = "2021-01-01 00:00"
TRAIN_END = "2023-12-31 23:00"
VAL_START = "2024-01-01 00:00"
VAL_END = "2024-12-31 23:00"
TEST_START = "2025-01-01 00:00"
TEST_END = "2025-12-31 23:00"

DEFAULT_ORDER = (1, 0, 1)
DEFAULT_SEASONAL_ORDER = (1, 0, 1, 24)
DEFAULT_TREND = "c"
DEFAULT_HORIZON = 24
DEFAULT_STEP = 24
DEFAULT_MAXITER = 100
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
