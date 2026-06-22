"""
DLinear config — theo Weather Forecasting Shared Protocol.
Xem PROTOCOL.md để biết chi tiết các ràng buộc.
"""

import os

# -- Đường dẫn ------------------------------------------
_DLIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # DLinear/

OUTPUT_DIR = os.path.join(_DLIN_DIR, "outputs")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data.npz")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, "feature_names.json")
TARGET_NAMES_PATH = os.path.join(OUTPUT_DIR, "target_names.json")

CHECKPOINT_DIR = os.path.join(_DLIN_DIR, "checkpoints")
CHECKPOINT_DIR_I = os.path.join(_DLIN_DIR, "checkpoints_i")

# -- DATA_PATH -------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(_DLIN_DIR), "weather_dataset.csv")

# -- Protocol: Target variables (7) ----------------------
# temp, rh, dew, pressure, cloud, wx, wy
TARGET_FEATURES = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'surface_pressure (hPa)',
    'cloud_cover (%)',
    'wx',
    'wy',
]

# -- Protocol: Input features (11) -----------------------
# 7 weather + 4 cyclical time encodings (precipitation removed by team decision)
INPUT_FEATURES = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'surface_pressure (hPa)',
    'cloud_cover (%)',
    'wx',
    'wy',
    'sin_hour',
    'cos_hour',
    'sin_doy',
    'cos_doy',
]

# -- Protocol constants (DO NOT EDIT without updating PROTOCOL.md) --
PROTOCOL_LOOKBACK = 120    # 5 days
PROTOCOL_FORECAST = 24     # 24 hours
PROTOCOL_TRAIN_YEARS = [2021, 2022, 2023]
PROTOCOL_VAL_YEAR = 2024
PROTOCOL_TEST_YEAR = 2025

# -- DLinear Architecture -------------------------------
MODEL_TYPE = "DLinear-S"
KERNEL_SIZE = 25

OUTPUT_DIR_S = os.path.join(OUTPUT_DIR, "S")
OUTPUT_DIR_I = os.path.join(OUTPUT_DIR, "I")


def get_checkpoint_dir(model_type):
    """Return checkpoint dir for a given model type."""
    return CHECKPOINT_DIR_I if model_type == "DLinear-I" else CHECKPOINT_DIR


def get_output_dir(model_type):
    """Return output dir for a given model type."""
    d = OUTPUT_DIR_I if model_type == "DLinear-I" else OUTPUT_DIR_S
    os.makedirs(d, exist_ok=True)
    return d


def is_individual(model_type):
    """Return individual flag for model construction."""
    return model_type == "DLinear-I"

# -- Time Series Windows (must match protocol) ----------
LOOKBACK = PROTOCOL_LOOKBACK
FORECAST = PROTOCOL_FORECAST

# -- Training -------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 1e-5

# -- Device ---------------------------------------------
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# -- Seed -----------------------------------------------
SEED = 42


# ============================================================
# Protocol self-check — chạy khi import để bắt vi phạm sớm
# ============================================================
def _validate_protocol():
    """Kiểm tra protocol — raise AssertionError nếu vi phạm."""
    # Feature counts
    assert len(INPUT_FEATURES) == 11, \
        f"Protocol requires 11 input features, got {len(INPUT_FEATURES)}"
    assert len(TARGET_FEATURES) == 7, \
        f"Protocol requires 7 target variables, got {len(TARGET_FEATURES)}"

    # All targets must be in inputs
    missing = [t for t in TARGET_FEATURES if t not in INPUT_FEATURES]
    assert not missing, f"Targets not in inputs: {missing}"

    # Time features must be last 4
    time_feats = ['sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']
    assert INPUT_FEATURES[-4:] == time_feats, \
        f"Last 4 input features must be {time_feats}"

    # Window params
    assert LOOKBACK == PROTOCOL_LOOKBACK, \
        f"LOOKBACK ({LOOKBACK}) != protocol ({PROTOCOL_LOOKBACK})"
    assert FORECAST == PROTOCOL_FORECAST, \
        f"FORECAST ({FORECAST}) != protocol ({PROTOCOL_FORECAST})"

    # Model type
    assert MODEL_TYPE in ("DLinear-S", "DLinear-I"), \
        f"Unknown MODEL_TYPE: {MODEL_TYPE}"

_validate_protocol()
