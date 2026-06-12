from __future__ import annotations

from typing import Iterable

import pandas as pd


def parse_order(value: str, expected_len: int, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse_error(label, value) from exc

    if len(parsed) != expected_len:
        raise argparse_error(label, value)
    if any(part < 0 for part in parsed):
        raise argparse_error(label, value)
    return parsed


def argparse_error(label: str, value: str) -> ValueError:
    return ValueError(
        f"{label} must be comma-separated non-negative integers, got {value!r}"
    )


def fit_sarima_models(
    train_targets: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    maxiter: int,
) -> dict[str, object]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    models: dict[str, object] = {}
    for variable in train_targets.columns:
        series = train_targets[variable].astype(float)
        print(f"  fitting {variable}...")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
            initialization="approximate_diffuse",
        )
        models[variable] = model.fit(disp=False, maxiter=maxiter)
    return models
