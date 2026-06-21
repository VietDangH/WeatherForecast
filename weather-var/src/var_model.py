"""VAR core: ADF stationarity test, differencing, AIC lag selection, fit."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

import config


def adf_pvalues(df: pd.DataFrame) -> dict:
    """ADF p-value per column. p > 0.05 => non-stationary => needs differencing."""
    return {c: float(adfuller(df[c].dropna())[1]) for c in df.columns}


def is_stationary(df: pd.DataFrame, alpha: float = 0.05) -> bool:
    """True only if every column rejects the unit-root null."""
    return all(p <= alpha for p in adf_pvalues(df).values())


def difference(df: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    out = df.copy()
    for _ in range(d):
        out = out.diff()
    return out.dropna()


def select_lag(df_diff: pd.DataFrame, maxlags: int = config.MAXLAGS):
    """AIC lag-order selection on the (stationary) differenced series."""
    sel = VAR(df_diff).select_order(maxlags)
    p = int(sel.aic)
    return max(p, 1), sel


def fit(df_diff: pd.DataFrame, p: int):
    """Fit VAR(p) by OLS (deterministic — no seed needed)."""
    return VAR(df_diff).fit(p)
