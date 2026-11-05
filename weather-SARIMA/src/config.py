"""Configuration for the per-feature ARIMA-family weather pipeline.

The defining idea of this folder: every target gets its OWN candidate grid
(``PER_FEATURE_GRID``) rather than one global grid shared across all variables.
Each grid is allowed to contain any member of the ARIMA family -- pure AR, pure
MA, ARMA, ARIMA, or seasonal SARIMA -- so the data decides per variable whether
a seasonal (daily, s=24) term is worth it. Nothing is forced to be seasonal and
nothing is forced to order (2,0,1).

The seasonal period is the DAILY cycle (s=24). Annual seasonality is deliberately
NOT modelled as a second SARIMA period (s~8766 is infeasible for SARIMAX and the
annual cycle is deterministic, not a unit root); it would enter, if at all, as an
exogenous Fourier term -- out of scope for this baseline.
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent              # weather-SARIMA/
REPO_ROOT = PKG_ROOT.parent             # d:\WeatherForecast
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "weather_dataset.csv"
DEFAULT_RESULTS_DIR = PKG_ROOT / "results"


# ---------------------------------------------------------------------------
# Data handling (identical to the shared PROTOCOL so results are comparable)
# ---------------------------------------------------------------------------
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

EXPECTED_SPLIT_ROWS = {"train": 26_280, "val": 8_784, "test": 8_760}

TRAIN_START = "2021-01-01 00:00"
TRAIN_END = "2023-12-31 23:00"
VAL_START = "2024-01-01 00:00"
VAL_END = "2024-12-31 23:00"
TEST_START = "2025-01-01 00:00"
TEST_END = "2025-12-31 23:00"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
SEASONAL_PERIOD = 24
DEFAULT_HORIZON = 24
DEFAULT_STEP = 24
DEFAULT_MAXITER = 100
DEFAULT_TREND = "c"
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# PER-FEATURE candidate grids
# ---------------------------------------------------------------------------
# Each entry is a list of (order, seasonal_order) candidates for THAT variable.
#   order          = (p, d, q)
#   seasonal_order = (P, D, Q, s);  (0, 0, 0, 0) == NO seasonal term
#
# d = D = 0 everywhere: ADF + KPSS agree the targets are level-stationary (the
# annual cycle is deterministic, not a unit root), so differencing is not used.
#
# The candidates below are bounded by each variable's own ACF/PACF signature
# (measured on the 2021-2023 train split). They are a STARTING SEARCH SPACE,
# not a decision -- validation MAE picks the winner per variable, and a
# non-seasonal / low-order model is free to win.
S = SEASONAL_PERIOD

PER_FEATURE_GRID: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]] = {
    # Very persistent, slow ACF decay, clear negative PACF spike at lag 24
    # -> high AR base + seasonal AR.
    "temp": [
        ((2, 0, 0), (1, 0, 0, S)),
        ((3, 0, 0), (1, 0, 0, S)),
        ((2, 0, 1), (1, 0, 1, S)),
        ((3, 0, 1), (1, 0, 1, S)),
    ],
    # Moderate persistence, faster ACF decay -> modest AR, seasonal AR/MA.
    "rh": [
        ((2, 0, 0), (1, 0, 0, S)),
        ((2, 0, 1), (1, 0, 1, S)),
        ((2, 0, 0), (0, 0, 1, S)),
        ((1, 0, 1), (1, 0, 1, S)),
    ],
    # Extremely persistent (ACF barely decays) but PACF~0 at lag 24
    # -> seasonal MA rather than seasonal AR; also try no seasonal.
    "dew": [
        ((2, 0, 0), (0, 0, 1, S)),
        ((3, 0, 0), (0, 0, 1, S)),
        ((2, 0, 1), (1, 0, 1, S)),
        ((2, 0, 0), (0, 0, 0, 0)),
    ],
    # Strongest seasonal PACF spike (-0.27); historically convergence-fussy
    # -> seasonal AR, enforce_stationarity on (see PER_FEATURE_OPTIONS).
    "pressure": [
        ((2, 0, 0), (1, 0, 0, S)),
        ((1, 0, 1), (1, 0, 0, S)),
        ((2, 0, 1), (1, 0, 1, S)),
        ((1, 0, 1), (1, 0, 1, S)),
    ],
    # Noisiest target, weak daily cycle (acf24~0.42, pacf24~0)
    # -> low AR + MA, seasonal optional (incl. fully non-seasonal ARMA).
    "cloud": [
        ((1, 0, 1), (0, 0, 0, 0)),
        ((2, 0, 0), (0, 0, 1, S)),
        ((2, 0, 1), (0, 0, 1, S)),
        ((1, 0, 1), (0, 0, 1, S)),
    ],
    # Moderate persistence and a real daily cycle -> AR + seasonal AR/MA.
    "wx": [
        ((2, 0, 0), (1, 0, 0, S)),
        ((2, 0, 1), (1, 0, 1, S)),
        ((2, 0, 0), (0, 0, 1, S)),
        ((1, 0, 1), (1, 0, 1, S)),
    ],
    # Weakest seasonality of all (acf24~0.38, pacf24~0)
    # -> likely non-seasonal ARMA; seasonal MA offered as an alternative.
    "wy": [
        ((1, 0, 1), (0, 0, 0, 0)),
        ((2, 0, 0), (0, 0, 0, 0)),
        ((2, 0, 0), (0, 0, 1, S)),
        ((2, 0, 1), (0, 0, 1, S)),
    ],
}

# Per-feature SARIMAX overrides. Anything not listed uses DEFAULT_FIT_OPTIONS.
DEFAULT_FIT_OPTIONS = {
    "trend": DEFAULT_TREND,
    "enforce_stationarity": False,
    "enforce_invertibility": False,
    "initialization": "approximate_diffuse",
    # Filter-only fit (skip the Kalman smoother's large (k,k,n) arrays). Keeps
    # .aic/.bic/.resid/.forecast/.append -- everything selection, diagnostics,
    # and rolling evaluation need -- while fitting in a fraction of the memory.
    # Essential on memory-limited machines; harmless elsewhere.
    "low_memory": True,
}

PER_FEATURE_OPTIONS: dict[str, dict] = {
    # pressure failed to converge with free roots in earlier runs; constrain it.
    "pressure": {"enforce_stationarity": True, "enforce_invertibility": True},
}


def fit_options_for(variable: str) -> dict:
    """Merge the default SARIMAX options with any per-feature override."""
    options = dict(DEFAULT_FIT_OPTIONS)
    options.update(PER_FEATURE_OPTIONS.get(variable, {}))
    return options
