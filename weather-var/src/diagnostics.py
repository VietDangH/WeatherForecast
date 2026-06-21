"""Post-fit diagnostics: stability, Ljung-Box, residual normality."""
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def run(res, lb_lag: int = 24):
    """Return (report_text, stable_flag, ljungbox_ok_flag)."""
    lines = []

    # --- Stability: all characteristic roots inside the unit circle ----------
    stable = bool(res.is_stable())
    lines.append(f"Stability (roots inside unit circle): {stable}")

    # --- Ljung-Box: residuals should be white noise (p > 0.05) ---------------
    resid = pd.DataFrame(res.resid)
    lines.append(f"Ljung-Box residual autocorrelation at lag {lb_lag} "
                 f"(p > 0.05 = white noise = good):")
    lb_ok = True
    for col in resid.columns:
        p = float(acorr_ljungbox(resid[col], lags=[lb_lag],
                                 return_df=True)["lb_pvalue"].iloc[0])
        flag = "OK" if p > 0.05 else "AUTOCORR LEFT"
        if p <= 0.05:
            lb_ok = False
        lines.append(f"  {col:>10}: p = {p:.4g}  [{flag}]")

    # --- Residual normality (affects validity of analytical CIs) -------------
    try:
        nt = res.test_normality()
        lines.append(f"Residual normality (joint): stat = {nt.test_statistic:.4g}, "
                     f"p = {nt.pvalue:.4g}")
        if nt.pvalue <= 0.05:
            lines.append("  -> non-normal residuals: analytical forecast intervals "
                         "are approximate.")
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"Residual normality: unavailable ({exc})")

    return "\n".join(lines), stable, lb_ok
