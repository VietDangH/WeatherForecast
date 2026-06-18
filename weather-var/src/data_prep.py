"""Load, clean, feature-engineer (wx/wy) and chronologically split the dataset."""
import numpy as np
import pandas as pd

import config


def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = col.split(" (")[0].strip()
        mapping[col] = config.PREFIX_RENAME.get(key, key)
    return df.rename(columns=mapping)


def load_clean() -> pd.DataFrame:
    """Return a clean, hourly-indexed dataframe with engineered wx/wy columns."""
    df = pd.read_csv(config.DATA_PATH)
    df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = _clean_names(df)

    # Data is already hourly; enforce the frequency and fill any incidental gap.
    df = df.asfreq("h")
    if df.isnull().any().any():
        df = df.interpolate("time").ffill().bfill()

    # Spec's -9999 sensor-error clip; harmless on this dataset (no negatives).
    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].clip(lower=0)

    # Wind decomposition: circular (speed, direction) -> Cartesian (wx, wy).
    rad = np.radians(df["wind_dir"])
    df["wx"] = df["wind_speed"] * np.cos(rad)
    df["wy"] = df["wind_speed"] * np.sin(rad)

    # Keep only the columns the project uses; drop any unused raw columns
    # (precipitation, soil moisture, gusts, low/mid/high cloud, etc.).
    keep = ["temp", "rh", "dew", "pressure", "cloud", "wind_speed", "wind_dir", "wx", "wy"]
    return df[[c for c in keep if c in df.columns]]


def split(df: pd.DataFrame):
    """Chronological split by year (never shuffled)."""
    train = df[df.index.year.isin(config.TRAIN_YEARS)]
    val = df[df.index.year.isin(config.VAL_YEARS)]
    test = df[df.index.year.isin(config.TEST_YEARS)]
    return train, val, test


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to the modeled target variables, in config order."""
    return df[config.TARGETS].copy()
