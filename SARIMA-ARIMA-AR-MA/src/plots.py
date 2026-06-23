"""Diagnostic plots: error-by-horizon, error-by-variable, predicted vs actual."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import ForecastResult


def write_plots(
    metrics: pd.DataFrame,
    result: ForecastResult,
    variables: list[str],
    plots_dir: Path,
    best_models: pd.DataFrame | None = None,
    max_points: int = 14 * 24,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    if best_models is not None and "label" in best_models.columns:
        labels = dict(zip(best_models["variable"], best_models["label"]))

    # Error accumulation over the horizon.
    horizon_metrics = metrics.loc[metrics["scope"] == "horizon"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(horizon_metrics["horizon"], horizon_metrics["mae"], marker="o", label="MAE")
    ax.plot(horizon_metrics["horizon"], horizon_metrics["rmse"], marker="o", label="RMSE")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Error")
    ax.set_title("Per-feature ARIMA error by horizon")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_horizon_error.png", dpi=160)
    plt.close(fig)

    # Error by variable.
    variable_metrics = metrics.loc[metrics["scope"] == "variable"].copy()
    x = np.arange(len(variable_metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, variable_metrics["mae"], width, label="MAE")
    ax.bar(x + width / 2, variable_metrics["rmse"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(variable_metrics["variable"])
    ax.set_ylabel("Error")
    ax.set_title("Per-feature ARIMA error by variable")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_variable_error.png", dpi=160)
    plt.close(fig)

    # Predicted vs actual per variable (first `max_points` hours).
    flat_times = pd.to_datetime(result.target_times.reshape(-1))
    for vi, variable in enumerate(variables):
        y_true = result.actuals[:, :, vi].reshape(-1)
        y_pred = result.predictions[:, :, vi].reshape(-1)
        limit = min(max_points, len(y_true))

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(flat_times[:limit], y_true[:limit], label="Actual", linewidth=1.7)
        ax.plot(flat_times[:limit], y_pred[:limit], label="Predicted", linewidth=1.3)
        title = f"Predicted vs actual: {variable}"
        if variable in labels:
            title += f"  [{labels[variable]}]"
        ax.set_title(title)
        ax.set_xlabel("Target time")
        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / f"predicted_vs_actual_{variable}.png", dpi=160)
        plt.close(fig)
