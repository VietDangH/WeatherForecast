from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_pre_fit_diagnostics(train_targets: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for variable in train_targets.columns:
        result = adfuller(train_targets[variable].astype(float), autolag="AIC")
        rows.append(
            {
                "variable": variable,
                "adf_stat": result[0],
                "p_value": result[1],
                "used_lag": result[2],
                "nobs": result[3],
                "stationary_at_0_05": result[1] < 0.05,
            }
        )
    return pd.DataFrame(rows)


def run_post_fit_diagnostics(models: dict[str, Any], lb_lag: int = 24) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    rows = []
    for variable, result in models.items():
        residuals = pd.Series(result.resid).replace([np.inf, -np.inf], np.nan).dropna()

        lb_stat = np.nan
        lb_p_value = np.nan
        if len(residuals) > lb_lag:
            lb = acorr_ljungbox(residuals, lags=[lb_lag], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p_value = float(lb["lb_pvalue"].iloc[0])

        jb_stat, jb_p_value, skew, kurtosis = jarque_bera(residuals)
        mle_retvals = getattr(result, "mle_retvals", {}) or {}

        rows.append(
            {
                "variable": variable,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "converged": bool(mle_retvals.get("converged", False)),
                "ljung_box_lag": lb_lag,
                "ljung_box_stat": lb_stat,
                "ljung_box_p_value": lb_p_value,
                "ljung_box_good_at_0_05": bool(lb_p_value > 0.05)
                if not np.isnan(lb_p_value)
                else False,
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_p_value": float(jb_p_value),
                "residual_normality_good_at_0_05": bool(jb_p_value > 0.05),
                "residual_skew": float(skew),
                "residual_kurtosis": float(kurtosis),
            }
        )
    return pd.DataFrame(rows)


def format_diagnostics(
    adf_diagnostics: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    fit_scope: str,
) -> str:
    lines = [
        "SARIMA diagnostics",
        "==================",
        "",
        f"Model: SARIMA(order={order}, seasonal_order={seasonal_order}, trend={trend!r})",
        f"Fit scope: {fit_scope}",
        "Targets are modeled independently and combined into a 24 x 7 forecast.",
        "",
        "ADF stationarity on train only",
        "-----------------------------",
        "For ADF, p < 0.05 rejects the unit-root null and is treated as stationary.",
        adf_diagnostics.to_string(index=False),
        "",
        "Post-fit diagnostics",
        "--------------------",
        "For Ljung-Box and Jarque-Bera, p > 0.05 is treated as good.",
        model_diagnostics.to_string(index=False),
        "",
    ]
    return "\n".join(lines)
