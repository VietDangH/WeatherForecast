from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import ForecastResult


def write_plots(
    metrics: pd.DataFrame,
    result: ForecastResult,
    variables: list[str],
    plots_dir: Path,
    max_points: int = 14 * 24,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    horizon_metrics = metrics.loc[metrics["scope"] == "horizon"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(horizon_metrics["horizon"], horizon_metrics["mae"], marker="o", label="MAE")
    ax.plot(horizon_metrics["horizon"], horizon_metrics["rmse"], marker="o", label="RMSE")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Error")
    ax.set_title("SARIMA error by horizon")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_horizon_error.png", dpi=160)
    plt.close(fig)

    variable_metrics = metrics.loc[metrics["scope"] == "variable"].copy()
    x = np.arange(len(variable_metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, variable_metrics["mae"], width, label="MAE")
    ax.bar(x + width / 2, variable_metrics["rmse"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(variable_metrics["variable"])
    ax.set_ylabel("Error")
    ax.set_title("SARIMA error by variable")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "per_variable_error.png", dpi=160)
    plt.close(fig)

    flat_times = pd.to_datetime(result.target_times.reshape(-1))
    for variable_index, variable in enumerate(variables):
        y_true = result.actuals[:, :, variable_index].reshape(-1)
        y_pred = result.predictions[:, :, variable_index].reshape(-1)
        limit = min(max_points, len(y_true))

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(flat_times[:limit], y_true[:limit], label="Actual", linewidth=1.7)
        ax.plot(flat_times[:limit], y_pred[:limit], label="Predicted", linewidth=1.3)
        ax.set_title(f"SARIMA predicted vs actual: {variable}")
        ax.set_xlabel("Target time")
        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plots_dir / f"predicted_vs_actual_{variable}.png", dpi=160)
        plt.close(fig)
