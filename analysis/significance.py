"""Forecast-accuracy significance testing across the five models.

Pairs every model's forecasts by (variable, calendar timestamp) over the 2025
test year, scores them against a single common set of actuals, and runs:

  * Diebold-Mariano (DM) pairwise tests (HAC/Newey-West variance), Holm-corrected,
    of each model vs. the univariate baseline (SARIMA) and vs. the best model;
  * a Model Confidence Set (MCS, Hansen-Lunde-Nahmias) via stationary bootstrap.

Loss = absolute error (consistent with the MAE headline). Per-variable tests use
each target's own unit; the "overall" test pools the seven targets after dividing
each error by that target's actual standard deviation (so units are comparable).

Why pair by timestamp (not by horizon): the pipelines roll on slightly different
phases, so a given hour is forecast at a (constant) different lead by each model.
Pooled over the year every model uses each lead equally often, so the mean loss
differential equals the overall-MAE gap; pairing on the shared weather only adds
power. This is stated as a caveat in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("D:/WeatherForecast")
TARGETS = ["temp", "rh", "dew", "pressure", "cloud", "wx", "wy"]
HORIZON = 24


# ---------------------------------------------------------------------------
# Reference actuals: every 2025 hour, per variable, from SARIMA's forecasts.csv
# (SARIMA forecasts each calendar hour exactly once, so y_true there is the
#  protocol-consistent ground truth for the whole test year).
# ---------------------------------------------------------------------------
_sar = pd.read_csv(ROOT / "SARIMA-ARIMA-AR-MA/results/forecasts.csv",
                   parse_dates=["target_time"])
REF = (_sar.pivot_table(index="target_time", columns="variable",
                        values="y_true", aggfunc="first")[TARGETS].sort_index())
REF_STD = REF.std()  # per-variable actual std, for overall standardisation


# ---------------------------------------------------------------------------
# Model loaders -> DataFrame(index=timestamp, columns=TARGETS) of y_pred
# ---------------------------------------------------------------------------
def load_sarima() -> pd.DataFrame:
    return (_sar.pivot_table(index="target_time", columns="variable",
                             values="y_pred", aggfunc="first")[TARGETS].sort_index())


def load_npz_with_time(path: Path) -> pd.DataFrame:
    """Models that saved origin_time (VAR, XGBoost)."""
    d = np.load(path, allow_pickle=True)
    yp, cols = d["y_pred"], list(d["targets"])
    ot = pd.to_datetime(d["origin_time"])
    N, H, V = yp.shape
    order = [cols.index(t) for t in TARGETS]
    times = (ot.values[:, None] +
             (np.arange(H) * np.timedelta64(1, "h"))[None, :]).ravel()
    df = pd.DataFrame(yp[:, :, order].reshape(N * H, V),
                      columns=TARGETS, index=pd.DatetimeIndex(times))
    return df[~df.index.duplicated(keep="first")].sort_index()


def _recover_origin_times(y_true: np.ndarray, cols: list[str]) -> np.ndarray:
    """Recover each origin block's horizon-1 timestamp by matching actuals to REF.

    y_true: (N, 24, V) real-scale actuals in `cols` order. Returns (N,) datetimes.
    """
    order = [cols.index(t) for t in TARGETS]
    yt = y_true[:, :, order]                       # (N, 24, 7) in TARGETS order
    rt = REF.index.values
    rv = REF.values                                # (T, 7)
    T = len(rt)

    # Hash on the first row's first 4 vars (rounded) for an O(N) candidate lookup.
    def key(vec):
        return tuple(np.round(vec[:4], 1))

    keymap: dict = {}
    for i in range(T - HORIZON + 1):
        keymap.setdefault(key(rv[i]), []).append(i)

    out = np.empty(yt.shape[0], dtype="datetime64[ns]")
    for o in range(yt.shape[0]):
        cands = keymap.get(key(yt[o, 0]), [])
        best, best_d = None, np.inf
        for ci in cands:
            d = np.abs(rv[ci:ci + HORIZON] - yt[o]).mean()
            if d < best_d:
                best_d, best = d, ci
        if best is None or best_d > 0.1:           # fallback: full slide search
            for ci in range(T - HORIZON + 1):
                d = np.abs(rv[ci:ci + HORIZON] - yt[o]).mean()
                if d < best_d:
                    best_d, best = d, ci
        if best_d > 0.25:
            raise RuntimeError(f"origin {o}: no confident match (L1={best_d:.3f})")
        out[o] = rt[best]
    return out


def load_arrays_matched(y_true: np.ndarray, y_pred: np.ndarray,
                        cols: list[str]) -> pd.DataFrame:
    """Models that saved only arrays without timestamps (LSTM, DLinear)."""
    ot = _recover_origin_times(y_true, cols)
    order = [cols.index(t) for t in TARGETS]
    N, H, V = y_pred.shape
    times = (ot[:, None] + (np.arange(H) * np.timedelta64(1, "h"))[None, :]).ravel()
    df = pd.DataFrame(y_pred[:, :, order].reshape(N * H, V),
                      columns=TARGETS, index=pd.DatetimeIndex(times))
    return df[~df.index.duplicated(keep="first")].sort_index()


def load_lstm() -> pd.DataFrame:
    base = ROOT / "weather-lstm/results_lstm"
    yt = np.load(base / "y_test_real.npy")
    yp = np.load(base / "ensemble_pred.npy")
    return load_arrays_matched(yt, yp, TARGETS)   # LSTM arrays already TARGETS order


def load_dlinear() -> pd.DataFrame:
    # DLinear stores raw target names but in the canonical TARGETS order, so
    # align positionally.
    d = np.load(ROOT / "DLinear/outputs/forecasts_eval.npz", allow_pickle=True)
    return load_arrays_matched(d["y_true"], d["y_pred"], TARGETS)


# ---------------------------------------------------------------------------
# Diebold-Mariano test (HAC / Newey-West variance, Harvey-Leybourne-Newbold corr.)
# ---------------------------------------------------------------------------
def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, hac_lag: int = HORIZON):
    """DM test on aligned loss series. dbar>0 => A has higher loss (A worse)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dbar = d.mean()
    d0 = d - dbar
    gamma0 = (d0 @ d0) / n
    var = gamma0
    for k in range(1, hac_lag + 1):
        cov = (d0[k:] @ d0[:n - k]) / n
        var += 2 * (1 - k / (hac_lag + 1)) * cov        # Bartlett weights
    var /= n
    if var <= 0:
        return dict(dbar=dbar, stat=np.nan, p=np.nan, n=n)
    stat = dbar / np.sqrt(var)
    # HLN small-sample correction factor (-> 1 for large n).
    corr = np.sqrt((n + 1 - 2 * hac_lag + hac_lag * (hac_lag - 1) / n) / n)
    stat_hln = stat * corr
    p = 2 * stats.t.sf(abs(stat_hln), df=n - 1)
    return dict(dbar=dbar, stat=stat_hln, p=p, n=n)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment. Returns name->adjusted p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(include_dlinear: bool) -> None:
    loaders = {
        "SARIMA": load_sarima,
        "VAR": lambda: load_npz_with_time(ROOT / "weather-var/results/forecasts_eval.npz"),
        "XGBoost": lambda: load_npz_with_time(ROOT / "weather-xgb/results/forecasts_eval.npz"),
        "LSTM": load_lstm,
    }
    if include_dlinear:
        loaders["DLinear"] = load_dlinear

    preds = {}
    for name, fn in loaders.items():
        preds[name] = fn()
        print(f"[load] {name:8s} {preds[name].shape}  "
              f"{preds[name].index.min()} .. {preds[name].index.max()}")

    # Common timestamps across all models (intersection).
    common = None
    for df in preds.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.intersection(REF.index).sort_values()
    print(f"\n[align] common timestamps: {len(common)}")

    actual = REF.loc[common]
    models = list(preds)

    # Per-variable absolute-error series, and standardised pooled "overall" loss.
    abserr = {m: (preds[m].loc[common] - actual).abs() for m in models}  # DataFrame per model
    overall_loss = {m: (abserr[m] / REF_STD).mean(axis=1).values for m in models}  # (T,) per model

    # Sanity: pooled standardised loss mean ordering should track overall MAE.
    print("\n[overall] mean standardised loss (lower=better):")
    means = {m: float(np.mean(overall_loss[m])) for m in models}
    for m in sorted(means, key=means.get):
        print(f"   {m:8s} {means[m]:.4f}")
    best = min(means, key=means.get)
    print(f"   best = {best}")

    # ---- DM vs baseline (SARIMA) and vs best, on the overall standardised loss
    def dm_table(ref_model):
        raw = {}
        for m in models:
            if m == ref_model:
                continue
            raw[m] = dm_test(overall_loss[m], overall_loss[ref_model])
        adj = holm({m: r["p"] for m, r in raw.items()})
        return raw, adj

    print(f"\n[DM] each model vs BEST ({best}); dbar>0 => model worse than best:")
    raw_best, adj_best = dm_table(best)
    for m in sorted(raw_best, key=lambda x: means[x]):
        r = raw_best[m]
        print(f"   {m:8s} dbar={r['dbar']:+.4f}  DM={r['stat']:+.2f}  "
              f"p={r['p']:.2e}  p_holm={adj_best[m]:.2e}")

    print(f"\n[DM] each model vs SARIMA baseline; dbar<0 => model better than SARIMA:")
    raw_base, adj_base = dm_table("SARIMA")
    for m in sorted(raw_base, key=lambda x: means[x]):
        r = raw_base[m]
        print(f"   {m:8s} dbar={r['dbar']:+.4f}  DM={r['stat']:+.2f}  "
              f"p={r['p']:.2e}  p_holm={adj_base[m]:.2e}")

    # ---- Model Confidence Set on the overall standardised loss ----
    from arch.bootstrap import MCS
    L = pd.DataFrame({m: overall_loss[m] for m in models})
    mcs = MCS(L, size=0.10, reps=5000, block_size=HORIZON, method="R", seed=42)
    mcs.compute()
    print("\n[MCS] overall (size=0.10, stationary block bootstrap, block=24):")
    pv = mcs.pvalues.sort_values("Pvalue")
    for name, row in pv.iterrows():
        inc = "IN " if row["Pvalue"] > 0.10 else "out"
        print(f"   {name:8s} MCS p={row['Pvalue']:.3f}  [{inc}]")

    # ---- Per-variable MCS membership (each target in its own unit) ----
    print("\n[MCS] per-variable 90% set membership:")
    per_var_sets = {}
    for v in TARGETS:
        Lv = pd.DataFrame({m: abserr[m][v].values for m in models})
        mv = MCS(Lv, size=0.10, reps=2000, block_size=HORIZON, method="R", seed=42)
        mv.compute()
        inset = sorted([m for m in models if mv.pvalues.loc[m, "Pvalue"] > 0.10],
                       key=lambda m: abserr[m][v].mean())
        best_v = min(models, key=lambda m: abserr[m][v].mean())
        per_var_sets[v] = inset
        print(f"   {v:9s} best={best_v:8s} set={{{', '.join(inset)}}}")

    # ---- Persist a compact results table for the report ----
    out = ROOT / "analysis"
    rows = []
    for m in models:
        rows.append(dict(
            model=m,
            overall_std_loss=means[m],
            dm_stat_vs_best=raw_best[m]["stat"] if m != best else np.nan,
            p_vs_best_holm=adj_best[m] if m != best else np.nan,
            dm_stat_vs_sarima=raw_base[m]["stat"] if m != "SARIMA" else np.nan,
            p_vs_sarima_holm=adj_base[m] if m != "SARIMA" else np.nan,
            mcs_pvalue=float(mcs.pvalues.loc[m, "Pvalue"]),
            in_mcs_90=bool(mcs.pvalues.loc[m, "Pvalue"] > 0.10),
        ))
    res = pd.DataFrame(rows).sort_values("overall_std_loss")
    res.to_csv(out / "significance_results.csv", index=False)
    pd.DataFrame({"variable": TARGETS,
                  "mcs_set": [", ".join(per_var_sets[v]) for v in TARGETS]}
                 ).to_csv(out / "significance_per_variable.csv", index=False)
    print(f"\n[done] n_models={len(models)}  common_obs={len(common)}")
    print(f"       wrote {out/'significance_results.csv'}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main(include_dlinear="--with-dlinear" in sys.argv)
