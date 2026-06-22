"""Post-fit diagnostics: stability, Ljung-Box, residual normality (per variable)."""
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


def per_variable(res, lb_lag: int = 24) -> pd.DataFrame:
    """Per-residual-series diagnostics.

    For each modeled variable: Ljung-Box autocorrelation (white-noise check) and
    a Jarque-Bera normality test (skew/kurtosis based). Returns one row per
    variable so the table can be written straight to CSV.
    """
    resid = pd.DataFrame(res.resid)
    rows = []
    for col in resid.columns:
        lb_p = float(acorr_ljungbox(resid[col], lags=[lb_lag],
                                    return_df=True)["lb_pvalue"].iloc[0])
        jb_stat, jb_p, skew, kurt = jarque_bera(resid[col].values)
        rows.append({
            "variable": col,
            "ljungbox_p": lb_p,
            "white_noise": lb_p > 0.05,
            "skew": float(skew),
            "kurtosis": float(kurt),
            "jarque_bera_p": float(jb_p),
            "residual_normal": jb_p > 0.05,
        })
    return pd.DataFrame(rows)


def run(res, lb_lag: int = 24):
    """Return (report_text, stable_flag, ljungbox_ok_flag, per_variable_df)."""
    lines = []

    # --- Stability: all characteristic roots inside the unit circle ----------
    stable = bool(res.is_stable())
    lines.append(f"Stability (roots inside unit circle): {stable}")

    pv = per_variable(res, lb_lag)
    lb_ok = bool(pv["white_noise"].all())

    # --- Ljung-Box per variable: residuals should be white noise -------------
    lines.append(f"\nLjung-Box residual autocorrelation at lag {lb_lag} "
                 f"(p > 0.05 = white noise = good):")
    for _, r in pv.iterrows():
        flag = "OK" if r["white_noise"] else "AUTOCORR LEFT"
        lines.append(f"  {r['variable']:>10}: p = {r['ljungbox_p']:.4g}  [{flag}]")

    # --- Residual normality per variable (Jarque-Bera) -----------------------
    lines.append("\nResidual normality per variable (Jarque-Bera; p > 0.05 = normal):")
    for _, r in pv.iterrows():
        flag = "normal" if r["residual_normal"] else "non-normal"
        lines.append(f"  {r['variable']:>10}: p = {r['jarque_bera_p']:.4g}  "
                     f"skew={r['skew']:>6.2f} kurt={r['kurtosis']:>7.2f}  [{flag}]")

    # --- Joint normality (overall, Lutkepohl) --------------------------------
    try:
        nt = res.test_normality()
        lines.append(f"\nResidual normality (joint): stat = {nt.test_statistic:.4g}, "
                     f"p = {nt.pvalue:.4g}")
        if nt.pvalue <= 0.05:
            lines.append("  -> non-normal residuals: analytical forecast intervals are "
                         "approximate; prefer the bootstrap interval.")
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"Residual normality: unavailable ({exc})")

    return "\n".join(lines), stable, lb_ok, pv
