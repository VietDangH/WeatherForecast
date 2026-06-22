from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import xgb_model


def predict_test(ds, models, X_te) -> np.ndarray:
    """Real-scale test forecasts -> (N, 24, 7)."""
    return ds.denorm(xgb_model.predict(models, X_te))


def compute_metrics(preds: np.ndarray, actual: np.ndarray) -> dict:
    """MAE/RMSE overall, per-horizon (24) and per-variable (7)."""
    err = preds - actual
    ae, se = np.abs(err), err ** 2
    return dict(
        overall_mae=float(ae.mean()),
        overall_rmse=float(np.sqrt(se.mean())),
        horizon_mae=ae.mean(axis=(0, 2)),               # (24,)
        horizon_rmse=np.sqrt(se.mean(axis=(0, 2))),
        var_mae=ae.mean(axis=(0, 1)),                    # (7,)
        var_rmse=np.sqrt(se.mean(axis=(0, 1))),
    )


def bootstrap_overall(preds, actual, n=C.BOOTSTRAP_N, level=C.BOOTSTRAP_CI, seed=0):
    """Bootstrap 95% CI on overall MAE/RMSE by resampling origins (windows)."""
    err = preds - actual
    ae_origin = np.abs(err).mean(axis=(1, 2))            # (N,) per-origin MAE
    se_origin = (err ** 2).mean(axis=(1, 2))             # (N,) per-origin MSE
    rng = np.random.default_rng(seed)
    N = len(ae_origin)
    maes, rmses = np.empty(n), np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        maes[b] = ae_origin[idx].mean()
        rmses[b] = np.sqrt(se_origin[idx].mean())
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return dict(
        mae_lo=float(np.percentile(maes, lo)), mae_hi=float(np.percentile(maes, hi)),
        rmse_lo=float(np.percentile(rmses, lo)), rmse_hi=float(np.percentile(rmses, hi)),
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_per_horizon(metrics):
    h = np.arange(1, C.HORIZON + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h, metrics["horizon_mae"], "-o", ms=3, label="MAE")
    ax.plot(h, metrics["horizon_rmse"], "-s", ms=3, label="RMSE")
    ax.set_xlabel("forecast horizon (h)")
    ax.set_ylabel("error (real scale, var-averaged)")
    ax.set_title("Error accumulation over the 24 h horizon (2025 test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "per_horizon_error.png", dpi=120)
    plt.close(fig)


def plot_per_variable(metrics):
    x = np.arange(len(C.TARGETS))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.4
    ax.bar(x - width / 2, metrics["var_mae"], width, label="MAE")
    ax.bar(x + width / 2, metrics["var_rmse"], width, label="RMSE")
    ax.set_xticks(x, C.TARGETS)
    ax.set_ylabel("error (real scale, horizon-averaged)")
    ax.set_title("Per-variable error (2025 test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "per_variable_error.png", dpi=120)
    plt.close(fig)


def plot_pred_vs_actual(ds, te_origins, preds, actual):
    """Reconstruct the tiled 2025 forecast (non-overlapping 24 h) vs actual."""
    # target timestamps for each origin block, concatenated in time order
    pos = (te_origins[:, None] + 1 + np.arange(C.HORIZON)[None, :]).ravel()
    t = ds.index[pos]
    pred_series = preds.reshape(-1, len(C.TARGETS))
    act_series = actual.reshape(-1, len(C.TARGETS))

    # full year
    fig, axes = plt.subplots(len(C.TARGETS), 1, figsize=(13, 2.0 * len(C.TARGETS)),
                             sharex=True)
    for v, var in enumerate(C.TARGETS):
        axes[v].plot(t, act_series[:, v], color="black", lw=0.6, label="actual")
        axes[v].plot(t, pred_series[:, v], color="tab:red", lw=0.6, alpha=0.8,
                     label="forecast")
        axes[v].set_ylabel(var)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_title("Rolling 24 h forecast vs actual — full 2025 test year")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "pred_vs_actual_full.png", dpi=120)
    plt.close(fig)

    # zoom: first 10 days
    zoom = 10 * 24
    fig, axes = plt.subplots(len(C.TARGETS), 1, figsize=(12, 2.0 * len(C.TARGETS)),
                             sharex=True)
    for v, var in enumerate(C.TARGETS):
        axes[v].plot(t[:zoom], act_series[:zoom, v], color="black", lw=1.0,
                     label="actual")
        axes[v].plot(t[:zoom], pred_series[:zoom, v], color="tab:red", lw=1.0,
                     label="forecast")
        axes[v].set_ylabel(var)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_title("Rolling 24 h forecast vs actual — first 10 days of 2025")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "pred_vs_actual_zoom.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# metrics.csv
# --------------------------------------------------------------------------
def write_metrics_csv(metrics, boot, seed_summary):
    rows = []
    rows.append(dict(
        scope="overall", variable="ALL", horizon="ALL",
        MAE=metrics["overall_mae"], RMSE=metrics["overall_rmse"],
        MAE_ci_lo=boot["mae_lo"], MAE_ci_hi=boot["mae_hi"],
        RMSE_ci_lo=boot["rmse_lo"], RMSE_ci_hi=boot["rmse_hi"],
        MAE_seed_std=seed_summary["mae_std"], RMSE_seed_std=seed_summary["rmse_std"],
        n_seeds=seed_summary["n_seeds"],
    ))
    for h in range(C.HORIZON):
        rows.append(dict(scope="per_horizon", variable="ALL", horizon=h + 1,
                         MAE=float(metrics["horizon_mae"][h]),
                         RMSE=float(metrics["horizon_rmse"][h])))
    for v, var in enumerate(C.TARGETS):
        rows.append(dict(scope="per_variable", variable=var, horizon="ALL",
                         MAE=float(metrics["var_mae"][v]),
                         RMSE=float(metrics["var_rmse"][v])))
    pd.DataFrame(rows).to_csv(C.METRICS_CSV, index=False)
