"""Per-feature ARIMA-family model selection.

Each target searches its OWN candidate grid (config.PER_FEATURE_GRID). For every
candidate we fit a SARIMAX on the train split, score it by rolling-origin MAE on
the validation split, and keep the variable's best (AIC breaks ties). The family
label (AR / MA / ARMA / ARIMA / SARIMA / ...) is derived from the chosen orders
purely for reporting.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import PER_FEATURE_GRID, fit_options_for

MODEL_METADATA_FILENAME = "training_metadata.json"

Order = tuple[int, int, int]
SeasonalOrder = tuple[int, int, int, int]
Candidate = tuple[Order, SeasonalOrder]


# ---------------------------------------------------------------------------
# Family naming
# ---------------------------------------------------------------------------
def family_name(order: Order, seasonal_order: SeasonalOrder) -> str:
    """Human-readable ARIMA-family label for an (order, seasonal_order) pair."""
    p, d, q = order
    big_p, big_d, big_q, s = seasonal_order
    has_seasonal = s > 0 and (big_p > 0 or big_d > 0 or big_q > 0)

    if has_seasonal:
        return "SARIMA" if (d > 0 or big_d > 0) else "SARMA"
    if d > 0:
        return "ARIMA"
    if p > 0 and q > 0:
        return "ARMA"
    if p > 0:
        return "AR"
    if q > 0:
        return "MA"
    return "mean-only"


def model_label(order: Order, seasonal_order: SeasonalOrder) -> str:
    fam = family_name(order, seasonal_order)
    if seasonal_order[3] > 0 and any(seasonal_order[:3]):
        return f"{fam}{order}x{seasonal_order}"
    return f"{fam}{order}"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def build_spec(order: Order, seasonal_order: SeasonalOrder, options: dict) -> dict:
    """SARIMAX keyword arguments that fully specify the model (no data)."""
    return {
        "order": order,
        "seasonal_order": seasonal_order,
        "trend": options.get("trend", "c"),
        "enforce_stationarity": options.get("enforce_stationarity", False),
        "enforce_invertibility": options.get("enforce_invertibility", False),
        "initialization": options.get("initialization", "approximate_diffuse"),
    }


def fit_model(
    series: pd.Series,
    order: Order,
    seasonal_order: SeasonalOrder,
    maxiter: int,
    options: dict,
) -> object:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series.astype(float), **build_spec(order, seasonal_order, options))
    return model.fit(
        disp=False, maxiter=maxiter, low_memory=options.get("low_memory", True)
    )


def rolling_refilter_forecast(
    spec: dict,
    params: np.ndarray,
    base_history: np.ndarray,
    eval_values: np.ndarray,
    horizon: int,
    step: int,
    max_origins: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Memory-light rolling forecasts.

    For each origin we rebuild a SARIMAX on ``base_history`` + realised
    ``eval_values`` up to that origin and run the Kalman FILTER only
    (``low_memory=True``), then forecast. This avoids the smoother's large
    ``(k_states, k_states, nobs)`` arrays that make ``.append()`` blow up memory
    on long hourly series, while producing identical forecasts (verified).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    base_history = np.asarray(base_history, dtype=float)
    eval_values = np.asarray(eval_values, dtype=float)

    starts = list(range(0, len(eval_values) - horizon + 1, step))
    if max_origins is not None:
        starts = starts[:max_origins]

    preds = np.empty((len(starts), horizon), dtype=float)
    for i, start in enumerate(starts):
        endog = np.concatenate([base_history, eval_values[:start]])
        filtered = SARIMAX(endog, **spec).filter(params, low_memory=True)
        preds[i] = np.asarray(filtered.forecast(horizon), dtype=float)
    return preds, starts


def _rolling_validation_score(
    spec: dict,
    params: np.ndarray,
    train_values: np.ndarray,
    validation_series: pd.Series,
    horizon: int,
    step: int,
) -> tuple[float, float]:
    """Rolling-origin MAE/RMSE on validation, conditioning on true history.

    Memory-light: each origin re-filters train + realised validation history
    (see :func:`rolling_refilter_forecast`) rather than chaining ``.append()``.
    """
    if horizon <= 0 or step <= 0:
        raise ValueError("selection horizon and step must be positive.")
    if len(validation_series) < horizon:
        raise ValueError("validation set is shorter than the selection horizon.")

    val_values = validation_series.to_numpy(dtype=float)
    preds, starts = rolling_refilter_forecast(
        spec, params, train_values, val_values, horizon, step
    )
    abs_errors, squared_errors = [], []
    for i, start in enumerate(starts):
        actual = val_values[start : start + horizon]
        errors = preds[i] - actual
        abs_errors.append(np.abs(errors))
        squared_errors.append(errors**2)

    all_abs = np.concatenate(abs_errors)
    all_sq = np.concatenate(squared_errors)
    return float(all_abs.mean()), float(np.sqrt(all_sq.mean()))


def _converged(result: object) -> bool:
    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    return bool(mle_retvals.get("converged", False))


# ---------------------------------------------------------------------------
# Per-variable selection
# ---------------------------------------------------------------------------
def select_for_variable(
    variable: str,
    train_series: pd.Series,
    validation_series: pd.Series,
    final_series: pd.Series,
    candidates: Iterable[Candidate],
    maxiter: int,
    selection_horizon: int,
    selection_step: int,
) -> dict[str, Any]:
    """Fit/score every candidate for one variable and return its best model."""
    options = fit_options_for(variable)
    candidates = list(candidates)

    best_key: tuple[float, float] | None = None
    best: dict[str, Any] | None = None
    selection_rows: list[dict[str, Any]] = []
    successful = 0

    for order, seasonal_order in candidates:
        label = model_label(order, seasonal_order)
        print(f"  [{variable}] candidate {label}", flush=True)
        try:
            result = fit_model(
                train_series, order, seasonal_order, maxiter, options
            )
            spec = build_spec(order, seasonal_order, options)
            val_mae, val_rmse = _rolling_validation_score(
                spec,
                result.params,
                train_series.to_numpy(dtype=float),
                validation_series,
                selection_horizon,
                selection_step,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            selection_rows.append(
                {
                    "variable": variable,
                    "family": family_name(order, seasonal_order),
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "validation_mae": np.nan,
                    "validation_rmse": np.nan,
                    "train_aic": np.nan,
                    "train_bic": np.nan,
                    "converged": False,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"    -> failed: {exc}", flush=True)
            continue

        successful += 1
        aic = float(result.aic)
        selection_rows.append(
            {
                "variable": variable,
                "family": family_name(order, seasonal_order),
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "train_aic": aic,
                "train_bic": float(result.bic),
                "converged": _converged(result),
                "status": "ok",
                "error": "",
            }
        )
        print(f"    -> val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} aic={aic:.1f}", flush=True)

        candidate_key = (val_mae, aic)  # MAE primary, AIC tie-break
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best = {
                "order": order,
                "seasonal_order": seasonal_order,
                "result": result,
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
            }

    if best is None:
        raise RuntimeError(f"No candidate fit successfully for {variable!r}.")

    # Refit the selected orders on the final fit scope if it differs from train.
    if final_series.index.equals(train_series.index):
        final_result = best["result"]
    else:
        print(f"  [{variable}] refitting selected {model_label(best['order'], best['seasonal_order'])} on final scope", flush=True)
        final_result = fit_model(
            final_series, best["order"], best["seasonal_order"], maxiter, options
        )

    best_row = {
        "variable": variable,
        "family": family_name(best["order"], best["seasonal_order"]),
        "order": str(best["order"]),
        "seasonal_order": str(best["seasonal_order"]),
        "label": model_label(best["order"], best["seasonal_order"]),
        "selected_by": "validation_mae",
        "validation_mae": best["validation_mae"],
        "validation_rmse": best["validation_rmse"],
        "successful_candidates": successful,
        "total_candidates": len(candidates),
        "final_fit_rows": len(final_series),
        "final_aic": float(final_result.aic),
        "final_bic": float(final_result.bic),
        "final_converged": _converged(final_result),
        **{k: v for k, v in fit_options_for(variable).items() if k != "initialization"},
    }
    return {
        "variable": variable,
        "final_result": final_result,
        "best_row": best_row,
        "selection_rows": selection_rows,
    }


def select_all(
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    final_fit_targets: pd.DataFrame,
    grid: dict[str, list[Candidate]] | None,
    maxiter: int,
    selection_horizon: int,
    selection_step: int,
    n_jobs: int = 1,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run per-feature selection for all targets, optionally in parallel."""
    grid = grid or PER_FEATURE_GRID
    variables = list(train_targets.columns)

    def job(variable: str) -> dict[str, Any]:
        if variable not in grid:
            raise KeyError(f"No candidate grid defined for {variable!r}.")
        return select_for_variable(
            variable=variable,
            train_series=train_targets[variable].astype(float),
            validation_series=validation_targets[variable].astype(float),
            final_series=final_fit_targets[variable].astype(float),
            candidates=grid[variable],
            maxiter=maxiter,
            selection_horizon=selection_horizon,
            selection_step=selection_step,
        )

    if n_jobs and n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(delayed(job)(v) for v in variables)
        except ImportError:
            print("joblib not installed; falling back to serial selection.", flush=True)
            results = [job(v) for v in variables]
    else:
        results = [job(v) for v in variables]

    models: dict[str, object] = {}
    best_rows, selection_rows = [], []
    for res in results:
        models[res["variable"]] = res["final_result"]
        best_rows.append(res["best_row"])
        selection_rows.extend(res["selection_rows"])
        print(
            f"  selected {res['variable']}: {res['best_row']['label']} "
            f"(val_mae={res['best_row']['validation_mae']:.4f})",
            flush=True,
        )

    # Preserve target order.
    best_df = pd.DataFrame(best_rows).set_index("variable").loc[variables].reset_index()
    return models, best_df, pd.DataFrame(selection_rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def make_model_specs(models: dict[str, object]) -> dict[str, dict[str, Any]]:
    """Build {variable: {spec, params, base_history}} for memory-light evaluation.

    ``base_history`` is the series each model was fit on (its training endog);
    rolling evaluation re-filters this plus realised history per origin.
    """
    specs: dict[str, dict[str, Any]] = {}
    for variable, result in models.items():
        model = result.model
        options = fit_options_for(variable)
        specs[variable] = {
            "spec": build_spec(model.order, model.seasonal_order, options),
            "params": np.asarray(result.params, dtype=float),
            "base_history": np.asarray(model.endog, dtype=float).ravel(),
        }
    return specs


def _safe_filename(variable: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in variable)
    return f"{safe}.pkl"


def save_model_artifacts(
    models: dict[str, object],
    best_models: pd.DataFrame,
    selection_results: pd.DataFrame,
    results_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {}
    for variable, result in models.items():
        path = models_dir / _safe_filename(variable)
        if hasattr(result, "save"):
            result.save(str(path))
        else:
            with path.open("wb") as fh:
                pickle.dump(result, fh)
        model_paths[variable] = path.relative_to(results_dir).as_posix()

    best_models = best_models.copy()
    best_models["model_path"] = best_models["variable"].map(model_paths)
    best_models.to_csv(models_dir / "best_models.csv", index=False)
    selection_results.to_csv(models_dir / "model_selection.csv", index=False)

    if metadata is not None:
        (models_dir / MODEL_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


def _load_result(path: Path) -> object:
    try:
        from statsmodels.iolib.smpickle import load_pickle

        return load_pickle(str(path))
    except ImportError:
        with path.open("rb") as fh:
            return pickle.load(fh)


def load_model_artifacts(
    results_dir: Path,
    variables: Iterable[str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, Any]]:
    models_dir = results_dir / "models"
    best_path = models_dir / "best_models.csv"
    if not best_path.exists():
        raise FileNotFoundError(
            f"No saved models at {best_path}. Run run.py (or train) first."
        )

    best_models = pd.read_csv(best_path)
    if {"variable", "model_path"} - set(best_models.columns):
        raise ValueError(f"{best_path} missing 'variable'/'model_path' columns.")

    load_variables = (
        list(variables) if variables is not None else best_models["variable"].tolist()
    )
    indexed = best_models.set_index("variable", drop=False)
    missing = [v for v in load_variables if v not in indexed.index]
    if missing:
        raise ValueError("Saved models missing targets: " + ", ".join(missing))

    models: dict[str, object] = {}
    for variable in load_variables:
        path = results_dir / Path(str(indexed.loc[variable, "model_path"]))
        if not path.exists():
            raise FileNotFoundError(f"Saved model file not found: {path}")
        models[variable] = _load_result(path)

    metadata_path = models_dir / MODEL_METADATA_FILENAME
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return models, best_models, metadata
