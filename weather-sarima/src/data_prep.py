from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_SPLIT_ROWS,
    HOURLY_FREQ,
    RAW_PREFIXES,
    TARGET_COLUMNS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _find_column(columns: list[str], prefix: str) -> str:
    matches = [column for column in columns if column.startswith(prefix)]
    if not matches:
        raise KeyError(f"Could not find a CSV column starting with {prefix!r}")

    exact_matches = [column for column in matches if column == prefix]
    if exact_matches:
        return exact_matches[0]

    unit_matches = [
        column
        for column in matches
        if len(column) > len(prefix) and column[len(prefix)] in {" ", "("}
    ]
    if unit_matches:
        return unit_matches[0]

    if len(matches) > 1:
        raise KeyError(f"Column prefix {prefix!r} is ambiguous: {matches}")
    return matches[0]


def load_weather_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path, parse_dates=["time"])

    rename_map = {
        _find_column(list(raw.columns), raw_prefix): clean_name
        for raw_prefix, clean_name in RAW_PREFIXES.items()
    }
    renamed = raw.rename(columns=rename_map)

    clean_columns = ["time", *RAW_PREFIXES.values()]
    missing = [column for column in clean_columns if column not in renamed.columns]
    if missing:
        raise KeyError(f"Missing expected cleaned columns: {missing}")

    data = renamed[clean_columns].copy()
    data = data.set_index("time").sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.asfreq(HOURLY_FREQ)

    for column in ["wind_speed", "wind_gust"]:
        data[column] = data[column].clip(lower=0)

    data = data.interpolate(method="time").ffill().bfill()

    wind_radians = np.deg2rad(data["wind_dir"] % 360)
    data["wx"] = data["wind_speed"] * np.cos(wind_radians)
    data["wy"] = data["wind_speed"] * np.sin(wind_radians)

    if data[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Target columns still contain nulls after preprocessing.")

    return data


def split_by_protocol(data: pd.DataFrame) -> DataSplits:
    train = data.loc[TRAIN_START:TRAIN_END].copy()
    val = data.loc[VAL_START:VAL_END].copy()
    test = data.loc[TEST_START:TEST_END].copy()

    actual_rows = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    if actual_rows != EXPECTED_SPLIT_ROWS:
        raise ValueError(
            "Protocol split row counts do not match. "
            f"Expected {EXPECTED_SPLIT_ROWS}, got {actual_rows}."
        )

    return DataSplits(train=train, val=val, test=test)
