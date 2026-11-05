"""Rolling-origin evaluation over the test year and metric aggregation."""
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
    model_specs: dict[str, dict[str, Any]],
    pretest_update: pd.DataFrame | None,
    test_targets: pd.DataFrame,
    horizon: int,
    step: int,
    max_origins: int | None = None,
) -> ForecastResult:
    """Memory-light rolling-origin forecasts over the test window.

    ``model_specs`` maps each variable to ``{spec, params, base_history}`` (see
    ``models.make_model_specs``). For every origin each model is re-filtered on
    its base history + realised history up to the origin -- no ``.append()``,
    so the Kalman smoother's large arrays are never allocated.
    """
    from models import rolling_refilter_forecast

    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be positive.")
    if len(test_targets) < horizon:
        raise ValueError("test set is shorter than the requested horizon.")

    variables = list(model_specs.keys())
    starts = list(range(0, len(test_targets) - horizon + 1, step))
    if max_origins is not None:
        starts = starts[:max_origins]

    predictions = np.empty((len(starts), horizon, len(variables)), dtype=float)
    actuals = np.empty_like(predictions)
    target_times = np.empty((len(starts), horizon), dtype="datetime64[ns]")

    # Per-variable: base history = train(+val if conditioning) before the test
    # window; eval values = the test series itself.
    for variable_index, variable in enumerate(variables):
        info = model_specs[variable]
        base_history = info["base_history"]
        if pretest_update is not None and not pretest_update.empty:
            base_history = np.concatenate(
                [base_history, pretest_update[variable].to_numpy(dtype=float)]
            )
        eval_values = test_targets[variable].to_numpy(dtype=float)
        preds, used_starts = rolling_refilter_forecast(
            info["spec"], info["params"], base_history, eval_values,
            horizon, step, max_origins=max_origins,
        )
        predictions[:, :, variable_index] = preds

    origins: list[pd.Timestamp] = []
    for origin_index, start in enumerate(starts):
        window = test_targets.iloc[start : start + horizon]
        actuals[origin_index] = window[variables].to_numpy(dtype=float)
        target_times[origin_index] = window.index.to_numpy(dtype="datetime64[ns]")
        origins.append(window.index[0] - pd.Timedelta(hours=1))

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
    for h in range(errors.shape[1]):
        rows.append(
            {
                "scope": "horizon",
                "horizon": h + 1,
                "variable": "all",
                "mae": float(abs_errors[:, h, :].mean()),
                "rmse": float(np.sqrt(squared_errors[:, h, :].mean())),
            }
        )
    for vi, variable in enumerate(variables):
        rows.append(
            {
                "scope": "variable",
                "horizon": np.nan,
                "variable": variable,
                "mae": float(abs_errors[:, :, vi].mean()),
                "rmse": float(np.sqrt(squared_errors[:, :, vi].mean())),
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

    mae_samples = np.empty(samples)
    rmse_samples = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, n_origins, size=n_origins)
        sampled = errors[idx]
        mae_samples[i] = np.abs(sampled).mean()
        rmse_samples[i] = np.sqrt((sampled**2).mean())

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
        results_dir / "overall_metrics.csv", index=False
    )
    metrics.loc[metrics["scope"] == "horizon"].to_csv(
        results_dir / "per_horizon_metrics.csv", index=False
    )
    metrics.loc[metrics["scope"] == "variable"].to_csv(
        results_dir / "per_variable_metrics.csv", index=False
    )


def save_forecasts(
    result: ForecastResult, variables: list[str], output_path: Path
) -> None:
    records = []
    for origin_index, origin in enumerate(result.origins):
        for h in range(result.predictions.shape[1]):
            target_time = pd.Timestamp(result.target_times[origin_index, h])
            for vi, variable in enumerate(variables):
                y_pred = result.predictions[origin_index, h, vi]
                y_true = result.actuals[origin_index, h, vi]
                records.append(
                    {
                        "origin": origin,
                        "target_time": target_time,
                        "horizon": h + 1,
                        "variable": variable,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "error": y_pred - y_true,
                    }
                )
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
