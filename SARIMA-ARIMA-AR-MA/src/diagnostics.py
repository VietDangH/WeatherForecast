"""Pre-fit (ADF) and post-fit (Ljung-Box, Jarque-Bera) diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEASONAL_PERIOD


def run_pre_fit_diagnostics(train_targets: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for variable in train_targets.columns:
        stat, p_value, used_lag, nobs, *_ = adfuller(
            train_targets[variable].astype(float), autolag="AIC"
        )
        rows.append(
            {
                "variable": variable,
                "adf_stat": stat,
                "p_value": p_value,
                "used_lag": used_lag,
                "nobs": nobs,
                "stationary_at_0_05": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)


def run_post_fit_diagnostics(
    models: dict[str, object], lb_lag: int = SEASONAL_PERIOD
) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    rows = []
    for variable, result in models.items():
        residuals = pd.Series(result.resid).replace([np.inf, -np.inf], np.nan).dropna()

        lb_stat = lb_p = np.nan
        if len(residuals) > lb_lag:
            lb = acorr_ljungbox(residuals, lags=[lb_lag], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p = float(lb["lb_pvalue"].iloc[0])

        jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
        mle_retvals = getattr(result, "mle_retvals", {}) or {}

        rows.append(
            {
                "variable": variable,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "converged": bool(mle_retvals.get("converged", False)),
                "ljung_box_lag": lb_lag,
                "ljung_box_stat": lb_stat,
                "ljung_box_p_value": lb_p,
                "ljung_box_good_at_0_05": bool(lb_p > 0.05) if not np.isnan(lb_p) else False,
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_p_value": float(jb_p),
                "residual_normality_good_at_0_05": bool(jb_p > 0.05),
                "residual_skew": float(skew),
                "residual_kurtosis": float(kurtosis),
            }
        )
    return pd.DataFrame(rows)


def format_diagnostics(
    adf_diagnostics: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    best_models: pd.DataFrame,
    fit_scope: str,
) -> str:
    lines = [
        "Per-feature ARIMA-family diagnostics",
        "====================================",
        "",
        "Each target is modelled by an independent univariate model whose family",
        "(AR / MA / ARMA / ARIMA / SARIMA) is chosen per variable by validation MAE",
        "from that variable's own candidate grid. Seasonal period s = "
        f"{SEASONAL_PERIOD} (daily).",
        f"Fit scope: {fit_scope}",
        "Targets are combined into a 24 x 7 forecast block.",
        "",
        "Selected model per target",
        "-------------------------",
        best_models[
            [
                "variable",
                "family",
                "order",
                "seasonal_order",
                "validation_mae",
                "validation_rmse",
                "final_converged",
            ]
        ].to_string(index=False),
        "",
        "ADF stationarity on train only",
        "------------------------------",
        "For ADF, p < 0.05 rejects the unit-root null (treated as stationary).",
        adf_diagnostics.to_string(index=False),
        "",
        "Post-fit diagnostics",
        "--------------------",
        "For Ljung-Box and Jarque-Bera, p > 0.05 is treated as good.",
        model_diagnostics.to_string(index=False),
        "",
    ]
    return "\n".join(lines)
