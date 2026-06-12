from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    predictions: np.ndarray
    actuals: np.ndarray
    origins: list[pd.Timestamp]
    target_times: np.ndarray


def rolling_origin_forecast(
    models: dict[str, Any],
    pretest_update: pd.DataFrame | None,
    test_targets: pd.DataFrame,
    horizon: int,
    step: int,
    max_origins: int | None = None,
) -> ForecastResult:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if len(test_targets) < horizon:
        raise ValueError("test set is shorter than the requested horizon.")

    variables = list(models.keys())
    starts = list(range(0, len(test_targets) - horizon + 1, step))
    if max_origins is not None:
        starts = starts[:max_origins]

    active_models = {}
    for variable, result in models.items():
        if pretest_update is None or pretest_update.empty:
            active_models[variable] = result
        else:
            active_models[variable] = result.append(
                pretest_update[variable].astype(float),
                refit=False,
            )

    predictions = np.empty((len(starts), horizon, len(variables)), dtype=float)
    actuals = np.empty_like(predictions)
    target_times = np.empty((len(starts), horizon), dtype="datetime64[ns]")
    origins: list[pd.Timestamp] = []

    for origin_index, start in enumerate(starts):
        actual_window = test_targets.iloc[start : start + horizon]
        actuals[origin_index] = actual_window[variables].to_numpy(dtype=float)
        target_times[origin_index] = actual_window.index.to_numpy(dtype="datetime64[ns]")
        origins.append(actual_window.index[0] - pd.Timedelta(hours=1))

        for variable_index, variable in enumerate(variables):
            forecast = active_models[variable].forecast(steps=horizon)
            predictions[origin_index, :, variable_index] = np.asarray(
                forecast,
                dtype=float,
            )

        if origin_index < len(starts) - 1:
            update_end = min(start + step, len(test_targets))
            update_block = test_targets.iloc[start:update_end]
            for variable in variables:
                active_models[variable] = active_models[variable].append(
                    update_block[variable].astype(float),
                    refit=False,
                )

    return ForecastResult(
        predictions=predictions,
        actuals=actuals,
        origins=origins,
        target_times=target_times,
    )


def compute_metrics(result: ForecastResult, variables: list[str]) -> pd.DataFrame:
    errors = result.predictions - result.actuals
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    rows = [
        {
            "scope": "overall",
            "horizon": np.nan,
            "variable": "all",
            "mae": float(abs_errors.mean()),
            "rmse": float(np.sqrt(squared_errors.mean())),
        }
    ]

    for horizon_index in range(errors.shape[1]):
        rows.append(
            {
                "scope": "horizon",
                "horizon": horizon_index + 1,
                "variable": "all",
                "mae": float(abs_errors[:, horizon_index, :].mean()),
                "rmse": float(np.sqrt(squared_errors[:, horizon_index, :].mean())),
            }
        )

    for variable_index, variable in enumerate(variables):
        rows.append(
            {
                "scope": "variable",
                "horizon": np.nan,
                "variable": variable,
                "mae": float(abs_errors[:, :, variable_index].mean()),
                "rmse": float(np.sqrt(squared_errors[:, :, variable_index].mean())),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_overall_ci(
    result: ForecastResult,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    if samples <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n_origins = result.predictions.shape[0]
    errors = result.predictions - result.actuals

    mae_samples = np.empty(samples, dtype=float)
    rmse_samples = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        sample_origins = rng.integers(0, n_origins, size=n_origins)
        sampled_errors = errors[sample_origins]
        mae_samples[sample_index] = np.abs(sampled_errors).mean()
        rmse_samples[sample_index] = np.sqrt((sampled_errors**2).mean())

    alpha = (1.0 - confidence) / 2.0
    return {
        "mae_ci_low": float(np.quantile(mae_samples, alpha)),
        "mae_ci_high": float(np.quantile(mae_samples, 1.0 - alpha)),
        "rmse_ci_low": float(np.quantile(rmse_samples, alpha)),
        "rmse_ci_high": float(np.quantile(rmse_samples, 1.0 - alpha)),
    }


def save_metrics(metrics: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    metrics.loc[metrics["scope"] == "overall"].to_csv(
        results_dir / "overall_metrics.csv",
        index=False,
    )
    metrics.loc[metrics["scope"] == "horizon"].to_csv(
        results_dir / "per_horizon_metrics.csv",
        index=False,
    )
    metrics.loc[metrics["scope"] == "variable"].to_csv(
        results_dir / "per_variable_metrics.csv",
        index=False,
    )


def save_forecasts(
    result: ForecastResult,
    variables: list[str],
    output_path: Path,
) -> None:
    records = []
    for origin_index, origin in enumerate(result.origins):
        for horizon_index in range(result.predictions.shape[1]):
            target_time = pd.Timestamp(result.target_times[origin_index, horizon_index])
            for variable_index, variable in enumerate(variables):
                y_pred = result.predictions[
                    origin_index,
                    horizon_index,
                    variable_index,
                ]
                y_true = result.actuals[
                    origin_index,
                    horizon_index,
                    variable_index,
                ]
                records.append(
                    {
                        "origin": origin,
                        "target_time": target_time,
                        "horizon": horizon_index + 1,
                        "variable": variable,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "error": y_pred - y_true,
                    }
                )

    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
