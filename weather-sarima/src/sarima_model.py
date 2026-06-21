from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_METADATA_FILENAME = "training_metadata.json"


def parse_order(value: str, expected_len: int, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse_error(label, value) from exc

    if len(parsed) != expected_len:
        raise argparse_error(label, value)
    if any(part < 0 for part in parsed):
        raise argparse_error(label, value)
    return parsed


def parse_order_grid(
    value: str,
    expected_len: int,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    candidates = []
    seen = set()
    for raw_candidate in value.split(";"):
        candidate_text = raw_candidate.strip()
        if not candidate_text:
            continue

        candidate = parse_order(candidate_text, expected_len, label)
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    if not candidates:
        raise argparse_error(label, value)
    return tuple(candidates)


def argparse_error(label: str, value: str) -> ValueError:
    return ValueError(
        f"{label} must be comma-separated non-negative integers, got {value!r}"
    )


def fit_sarima_model(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    maxiter: int,
) -> object:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization="approximate_diffuse",
    )
    return model.fit(disp=False, maxiter=maxiter)


def fit_sarima_models(
    train_targets: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    maxiter: int,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for variable in train_targets.columns:
        series = train_targets[variable].astype(float)
        print(f"  fitting {variable}...")
        models[variable] = fit_sarima_model(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            maxiter=maxiter,
        )
    return models


def _rolling_validation_score(
    result: object,
    validation_series: pd.Series,
    horizon: int,
    step: int,
) -> tuple[float, float]:
    if horizon <= 0:
        raise ValueError("selection horizon must be positive.")
    if step <= 0:
        raise ValueError("selection step must be positive.")
    if len(validation_series) < horizon:
        raise ValueError("validation set is shorter than the selection horizon.")

    starts = list(range(0, len(validation_series) - horizon + 1, step))
    active_result = result
    abs_errors = []
    squared_errors = []

    for origin_index, start in enumerate(starts):
        actual_window = validation_series.iloc[start : start + horizon].astype(float)
        forecast = active_result.forecast(steps=horizon)
        errors = np.asarray(forecast, dtype=float) - actual_window.to_numpy(dtype=float)
        abs_errors.append(np.abs(errors))
        squared_errors.append(errors**2)

        is_last_origin = origin_index == len(starts) - 1
        if not is_last_origin:
            update_end = min(start + step, len(validation_series))
            active_result = active_result.append(
                validation_series.iloc[start:update_end].astype(float),
                refit=False,
            )

    all_abs_errors = np.concatenate(abs_errors)
    all_squared_errors = np.concatenate(squared_errors)
    return (
        float(all_abs_errors.mean()),
        float(np.sqrt(all_squared_errors.mean())),
    )


def _order_text(order: Iterable[int]) -> str:
    return ",".join(str(part) for part in order)


def _result_converged(result: object) -> bool:
    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    return bool(mle_retvals.get("converged", False))


def fit_best_sarima_models(
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    final_fit_targets: pd.DataFrame,
    order_candidates: Iterable[tuple[int, int, int]],
    seasonal_order_candidates: Iterable[tuple[int, int, int, int]],
    trend: str,
    maxiter: int,
    selection_horizon: int,
    selection_step: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    models: dict[str, object] = {}
    best_rows = []
    selection_rows = []

    order_candidates = tuple(order_candidates)
    seasonal_order_candidates = tuple(seasonal_order_candidates)
    total_candidates = len(order_candidates) * len(seasonal_order_candidates)

    for variable in train_targets.columns:
        print(f"  selecting {variable} across {total_candidates} candidate(s)...")
        train_series = train_targets[variable].astype(float)
        validation_series = validation_targets[variable].astype(float)
        final_series = final_fit_targets[variable].astype(float)

        best_key: tuple[float, float] | None = None
        best_order: tuple[int, int, int] | None = None
        best_seasonal_order: tuple[int, int, int, int] | None = None
        best_result: object | None = None
        best_validation_mae = np.nan
        best_validation_rmse = np.nan
        successful_candidates = 0

        for order in order_candidates:
            for seasonal_order in seasonal_order_candidates:
                order_label = _order_text(order)
                seasonal_order_label = _order_text(seasonal_order)
                print(
                    f"    candidate order=({order_label}) "
                    f"seasonal_order=({seasonal_order_label})"
                )

                try:
                    result = fit_sarima_model(
                        train_series,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=trend,
                        maxiter=maxiter,
                    )
                    validation_mae, validation_rmse = _rolling_validation_score(
                        result=result,
                        validation_series=validation_series,
                        horizon=selection_horizon,
                        step=selection_step,
                    )
                except Exception as exc:  # noqa: BLE001
                    selection_rows.append(
                        {
                            "variable": variable,
                            "order": order_label,
                            "seasonal_order": seasonal_order_label,
                            "trend": trend,
                            "selection_horizon": selection_horizon,
                            "selection_step": selection_step,
                            "validation_mae": np.nan,
                            "validation_rmse": np.nan,
                            "train_aic": np.nan,
                            "train_bic": np.nan,
                            "train_converged": False,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    print(f"      failed: {exc}")
                    continue

                successful_candidates += 1
                candidate_key = (validation_mae, float(result.aic))
                selection_rows.append(
                    {
                        "variable": variable,
                        "order": order_label,
                        "seasonal_order": seasonal_order_label,
                        "trend": trend,
                        "selection_horizon": selection_horizon,
                        "selection_step": selection_step,
                        "validation_mae": validation_mae,
                        "validation_rmse": validation_rmse,
                        "train_aic": float(result.aic),
                        "train_bic": float(result.bic),
                        "train_converged": _result_converged(result),
                        "status": "ok",
                        "error": "",
                    }
                )
                print(
                    f"      val_mae={validation_mae:.4f} "
                    f"val_rmse={validation_rmse:.4f}"
                )

                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_result = result
                    best_validation_mae = validation_mae
                    best_validation_rmse = validation_rmse

        if best_result is None or best_order is None or best_seasonal_order is None:
            raise RuntimeError(f"No SARIMA candidate fit successfully for {variable!r}.")

        if final_series.index.equals(train_series.index):
            final_result = best_result
        else:
            print(f"    refitting selected {variable} on final fit scope...")
            final_result = fit_sarima_model(
                final_series,
                order=best_order,
                seasonal_order=best_seasonal_order,
                trend=trend,
                maxiter=maxiter,
            )

        models[variable] = final_result
        best_rows.append(
            {
                "variable": variable,
                "order": _order_text(best_order),
                "seasonal_order": _order_text(best_seasonal_order),
                "trend": trend,
                "selected_by": "validation_mae",
                "selection_horizon": selection_horizon,
                "selection_step": selection_step,
                "validation_mae": best_validation_mae,
                "validation_rmse": best_validation_rmse,
                "successful_candidates": successful_candidates,
                "total_candidates": total_candidates,
                "final_fit_rows": len(final_series),
                "final_aic": float(final_result.aic),
                "final_bic": float(final_result.bic),
                "final_converged": _result_converged(final_result),
            }
        )
        print(
            f"    selected order=({_order_text(best_order)}) "
            f"seasonal_order=({_order_text(best_seasonal_order)}) "
            f"validation_mae={best_validation_mae:.4f}"
        )

    return (
        models,
        pd.DataFrame(best_rows),
        pd.DataFrame(selection_rows),
    )


def _safe_model_filename(variable: str) -> str:
    safe_variable = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in variable
    )
    return f"{safe_variable}.pkl"


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
        model_path = models_dir / _safe_model_filename(variable)
        if hasattr(result, "save"):
            result.save(str(model_path))
        else:
            with model_path.open("wb") as file:
                pickle.dump(result, file)
        model_paths[variable] = model_path.relative_to(results_dir).as_posix()

    best_models = best_models.copy()
    best_models["model_path"] = best_models["variable"].map(model_paths)
    best_models.to_csv(models_dir / "best_models.csv", index=False)
    selection_results.to_csv(models_dir / "model_selection.csv", index=False)

    if metadata is not None:
        metadata_path = models_dir / MODEL_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _load_model_result(model_path: Path) -> object:
    try:
        from statsmodels.iolib.smpickle import load_pickle
    except ImportError:
        with model_path.open("rb") as file:
            return pickle.load(file)

    return load_pickle(str(model_path))


def load_model_artifacts(
    results_dir: Path,
    variables: Iterable[str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, Any]]:
    models_dir = results_dir / "models"
    best_models_path = models_dir / "best_models.csv"
    if not best_models_path.exists():
        raise FileNotFoundError(
            f"No saved SARIMA models found at {best_models_path}. "
            "Run train.py before predict.py."
        )

    best_models = pd.read_csv(best_models_path)
    required_columns = {"variable", "model_path"}
    missing_columns = required_columns.difference(best_models.columns)
    if missing_columns:
        raise ValueError(
            f"{best_models_path} is missing required columns: {sorted(missing_columns)}"
        )

    if variables is None:
        load_variables = best_models["variable"].tolist()
    else:
        load_variables = list(variables)

    best_by_variable = best_models.set_index("variable", drop=False)
    missing_variables = [
        variable for variable in load_variables if variable not in best_by_variable.index
    ]
    if missing_variables:
        raise ValueError(
            "Saved models are missing target variables: "
            + ", ".join(missing_variables)
        )

    models: dict[str, object] = {}
    for variable in load_variables:
        relative_model_path = Path(str(best_by_variable.loc[variable, "model_path"]))
        model_path = results_dir / relative_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model file not found: {model_path}")
        models[variable] = _load_model_result(model_path)

    metadata_path = models_dir / MODEL_METADATA_FILENAME
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return models, best_models, metadata
