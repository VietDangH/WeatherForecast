"""Rolling-origin 24h forecasting across the test year.

Fit VAR ONCE on train. For each origin t in the test year, condition on the
true differenced history up to t, forecast t+1..t+H, invert differencing (and
the scaler if used), and score against the real-scale actuals.
"""
import numpy as np
import pandas as pd

import config


def model_space(scaled, d):
    """Series the VAR is fit/forecast in: levels (d=0) or differences (d>=1).

    Returns an array aligned to the full index with the first ``d`` rows NaN so
    positional slicing matches the real series.
    """
    if d == 0:
        return scaled.copy()
    series = np.full_like(scaled, np.nan, dtype=float)
    series[d:] = np.diff(scaled, n=d, axis=0)
    return series


def invert(fc, last_scaled, scaler, d):
    """Map a forecast from model space back to the real scale."""
    fc_scaled = fc if d == 0 else last_scaled + np.cumsum(fc, axis=0)
    if scaler is not None:
        return scaler.inverse_transform(fc_scaled)
    return fc_scaled


def rolling_origin(res, real_df, scaled_df, scaler, d=config.DIFF_ORDER,
                   horizon=config.HORIZON, step=config.ROLL_STEP):
    """Run the rolling evaluation.

    Parameters use the FULL series (all years) so the history at the first test
    origins can draw on the true tail of the prior year.
    """
    cols = list(real_df.columns)
    n = len(cols)
    real = real_df.values            # unscaled, real scale
    scaled = scaled_df.values        # scaled if STANDARDIZE else == real
    times = real_df.index
    years = times.year.values
    k = res.k_ar

    series = model_space(scaled, d)  # levels or differences, NaN-padded
    valid_start = d                  # first usable row in `series`

    abs_sum = np.zeros((horizon, n))
    sq_sum = np.zeros((horizon, n))
    count = 0

    preds, actuals, pred_times = [], [], []   # for contiguous plotting

    test_pos = np.where(np.isin(years, config.TEST_YEARS))[0]
    for i in test_pos[::step]:
        if i - k + 1 < valid_start or i + horizon >= len(real):
            continue
        hist = series[i - k + 1:i + 1]                 # last k rows up to t
        fc = res.forecast(hist, steps=horizon)
        fc_real = invert(fc, scaled[i], scaler, d)
        actual = real[i + 1:i + 1 + horizon]

        err = fc_real - actual
        abs_sum += np.abs(err)
        sq_sum += err ** 2
        count += 1

        preds.append(fc_real)
        actuals.append(actual)
        pred_times.append(times[i + 1:i + 1 + horizon])

    if count == 0:
        raise RuntimeError("No valid rolling origins produced.")

    metrics = _metrics(abs_sum, sq_sum, count, horizon, n, cols)

    # Contiguous prediction frame (meaningful when step == horizon).
    pred_index = pred_times[0].append([p for p in pred_times[1:]]) \
        if len(pred_times) > 1 else pred_times[0]
    pred_df = pd.DataFrame(np.vstack(preds), index=pred_index, columns=cols)
    actual_df = pd.DataFrame(np.vstack(actuals), index=pred_index, columns=cols)

    return metrics, pred_df, actual_df, count


def _metrics(abs_sum, sq_sum, count, horizon, n, cols):
    total = count * horizon * n
    overall = {
        "MAE": abs_sum.sum() / total,
        "RMSE": np.sqrt(sq_sum.sum() / total),
    }
    per_horizon = pd.DataFrame({
        "horizon": np.arange(1, horizon + 1),
        "MAE": abs_sum.sum(axis=1) / (count * n),
        "RMSE": np.sqrt(sq_sum.sum(axis=1) / (count * n)),
    })
    per_variable = pd.DataFrame({
        "variable": cols,
        "MAE": abs_sum.sum(axis=0) / (count * horizon),
        "RMSE": np.sqrt(sq_sum.sum(axis=0) / (count * horizon)),
    })
    return {"overall": overall, "per_horizon": per_horizon,
            "per_variable": per_variable}


def to_long_csv(metrics) -> pd.DataFrame:
    """Flatten metrics into a single long-format dataframe for metrics.csv."""
    rows = [("overall", "all", metrics["overall"]["MAE"], metrics["overall"]["RMSE"])]
    for _, r in metrics["per_horizon"].iterrows():
        rows.append(("horizon", int(r["horizon"]), r["MAE"], r["RMSE"]))
    for _, r in metrics["per_variable"].iterrows():
        rows.append(("variable", r["variable"], r["MAE"], r["RMSE"]))
    return pd.DataFrame(rows, columns=["scope", "key", "MAE", "RMSE"])
