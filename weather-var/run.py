"""Entry point: prep -> stationarity/diff -> fit -> diagnostics -> rolling eval.

Run from anywhere:  python weather-var/run.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import data_prep
import var_model
import diagnostics
import evaluate


def main():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    log = []

    def say(msg):
        print(msg)
        log.append(str(msg))

    # 1. Data prep -----------------------------------------------------------
    say("== Data preparation ==")
    df = data_prep.load_clean()
    train, val, test = data_prep.split(df)
    say(f"Targets: {config.TARGETS}")
    say(f"Rows  train={len(train)}  val={len(val)}  test={len(test)}")

    real = data_prep.targets(df)          # full series, real scale
    train_real = data_prep.targets(train)

    # 2. Optional standardization (fit on train only) ------------------------
    scaler = None
    scaled = real.copy()
    if config.STANDARDIZE:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train_real.values)
        scaled = pd.DataFrame(scaler.transform(real.values),
                              index=real.index, columns=real.columns)
        say("Standardized (scaler fit on train only).")

    train_scaled = scaled[scaled.index.year.isin(config.TRAIN_YEARS)]

    # 3. Stationarity + differencing ----------------------------------------
    say("\n== Stationarity (ADF on train) ==")
    adf_before = var_model.adf_pvalues(train_scaled)
    for c, p in adf_before.items():
        say(f"  {c:>10}: p = {p:.4g}  [{'stationary' if p <= 0.05 else 'NON-stationary'}]")

    d = 0
    train_diff = train_scaled
    if not var_model.is_stationary(train_scaled):
        d = config.DIFF_ORDER
        train_diff = var_model.difference(train_scaled, d)
        say(f"\nApplied differencing d={d}. Re-checking ADF:")
        for c, p in var_model.adf_pvalues(train_diff).items():
            say(f"  {c:>10}: p = {p:.4g}  [{'stationary' if p <= 0.05 else 'STILL non-stat'}]")
    else:
        say("\nSeries already stationary; d=0.")

    # 4. Lag selection + fit -------------------------------------------------
    say("\n== Lag-order selection (AIC) ==")
    full_diff = var_model.difference(scaled, d) if d else scaled
    train_diff_full = full_diff[full_diff.index.year.isin(config.TRAIN_YEARS)]
    p, sel = var_model.select_lag(train_diff_full, config.MAXLAGS)
    say(f"Selected lag order p = {p} (maxlags={config.MAXLAGS})")

    res = var_model.fit(train_diff_full, p)
    say(f"Fitted VAR({p}) on {len(train_diff_full)} differenced train rows, "
        f"{len(config.TARGETS)} variables.")

    # 5. Diagnostics ---------------------------------------------------------
    say("\n== Diagnostics ==")
    diag_text, stable, lb_ok = diagnostics.run(res, lb_lag=config.HORIZON)
    say(diag_text)
    if not stable:
        say("WARNING: VAR is not stable — forecasts may diverge.")
    if not lb_ok:
        say("WARNING: residual autocorrelation remains — consider larger maxlags.")

    # 6. Rolling-origin evaluation ------------------------------------------
    say("\n== Rolling-origin evaluation (test year) ==")
    metrics, pred_df, actual_df, n_origins = evaluate.rolling_origin(
        res, real, scaled, scaler, d=d,
        horizon=config.HORIZON, step=config.ROLL_STEP)
    say(f"Origins evaluated: {n_origins} (step={config.ROLL_STEP}h)")
    say(f"Overall  MAE = {metrics['overall']['MAE']:.4f}   "
        f"RMSE = {metrics['overall']['RMSE']:.4f}")
    say("\nPer-variable:")
    say(metrics["per_variable"].to_string(index=False))

    # 7. Persist results -----------------------------------------------------
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    evaluate.to_long_csv(metrics).to_csv(
        os.path.join(config.RESULTS_DIR, "metrics.csv"), index=False)

    with open(os.path.join(config.RESULTS_DIR, "diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(log))

    _plots(res, real, scaled, scaler, max(d, 1), metrics, pred_df, actual_df)
    say(f"\nResults written to {config.RESULTS_DIR}")


def _plots(res, real, scaled, scaler, d, metrics, pred_df, actual_df):
    cols = list(real.columns)

    # 7a. Predicted vs actual across the test year (one subplot per variable).
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.4 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        ax.plot(actual_df.index, actual_df[c], lw=0.7, label="actual")
        ax.plot(pred_df.index, pred_df[c], lw=0.7, label="pred")
        ax.set_ylabel(c)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_title("Rolling 24h forecast vs actual (test year)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "pred_vs_actual.png"), dpi=110)
    plt.close(fig)

    # 7b. Per-horizon error curve (error accumulation over the horizon).
    ph = metrics["per_horizon"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ph["horizon"], ph["MAE"], marker="o", label="MAE")
    ax.plot(ph["horizon"], ph["RMSE"], marker="s", label="RMSE")
    ax.set_xlabel("forecast horizon (h)")
    ax.set_ylabel("error (real scale)")
    ax.set_title("Error accumulation over the 24h horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "per_horizon_error.png"), dpi=110)
    plt.close(fig)

    # 7c. Analytical forecast-interval bands at the first test origin.
    times = real.index
    years = times.year.values
    test_pos = np.where(np.isin(years, config.TEST_YEARS))[0]
    k = res.k_ar
    i = next((j for j in test_pos if j - k + 1 >= d and j + config.HORIZON < len(real)), None)
    if i is not None:
        series = evaluate.model_space(scaled.values, d)
        hist = series[i - k + 1:i + 1]
        mid, lower, upper = res.forecast_interval(hist, steps=config.HORIZON, alpha=0.05)

        def inv(arr):
            return evaluate.invert(arr, scaled.values[i], scaler, d)

        m, lo, hi = inv(mid), inv(lower), inv(upper)

        # Bootstrap bands (residual resampling) — no normality assumption.
        bmid, blo, bhi, _ = evaluate.bootstrap_forecast_interval(
            res, hist, steps=config.HORIZON, n_boot=1000, alpha=0.05)
        bm, blo, bhi = inv(bmid), inv(blo), inv(bhi)

        actual = real.values[i + 1:i + 1 + config.HORIZON]
        vi = 0  # first target variable
        h = np.arange(1, config.HORIZON + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(h, actual[:, vi], marker="o", label="actual")
        ax.plot(h, m[:, vi], marker="x", label="forecast")
        ax.fill_between(h, lo[:, vi], hi[:, vi], alpha=0.25,
                        label="95% analytical (Gaussian)")
        ax.fill_between(h, blo[:, vi], bhi[:, vi], alpha=0.0,
                        edgecolor="C3", facecolor="none", hatch="///",
                        label="95% bootstrap (resampled residuals)")
        ax.plot(h, blo[:, vi], color="C3", lw=0.8)
        ax.plot(h, bhi[:, vi], color="C3", lw=0.8)
        ax.set_xlabel("forecast horizon (h)")
        ax.set_ylabel(cols[vi])
        ax.set_title(f"24h forecast — analytical vs bootstrap 95% interval — "
                     f"{cols[vi]} (first test origin)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(config.PLOTS_DIR, f"forecast_interval_{cols[vi]}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
