"""XGBoost model: one multi-output regressor per target variable.
"""
from __future__ import annotations
import json
import time
import numpy as np
from xgboost import XGBRegressor
from . import config as C


def _make_regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(
        random_state=seed,
        early_stopping_rounds=C.EARLY_STOPPING_ROUNDS,
        **C.XGB_PARAMS,
    )


def train_models(X_tr: np.ndarray, Y_tr: np.ndarray,
                 X_va: np.ndarray, Y_va: np.ndarray, seed: int):
    """Fit one multi-output regressor per target variable.

    Y_tr / Y_va : (N, 24, 7) scaled targets.
    Returns (models, history) where:
      models[var]  -> fitted XGBRegressor (engineered features -> 24)
      history[var] -> dict(val_rmse=[...per round], best_iteration, best_val_rmse,
                           train_rmse=final-train-rmse-scaled)
    """
    models, history = {}, {}
    for v, var in enumerate(C.TARGETS):
        ytr, yva = Y_tr[:, :, v], Y_va[:, :, v]            # (N, 24)
        model = _make_regressor(seed)
        t0 = time.time()
        model.fit(X_tr, ytr, eval_set=[(X_va, yva)], verbose=False)
        dt = time.time() - t0

        val_curve = model.evals_result()["validation_0"]["rmse"]
        # cheap final train RMSE (scaled) to expose the train/val gap
        tr_pred = model.predict(X_tr)
        train_rmse = float(np.sqrt(np.mean((tr_pred - ytr) ** 2)))

        models[var] = model
        history[var] = dict(
            val_rmse=val_curve,
            best_iteration=int(model.best_iteration),
            best_val_rmse=float(val_curve[model.best_iteration]),
            train_rmse=train_rmse,
            fit_seconds=round(dt, 1),
        )
    return models, history


def predict(models, X: np.ndarray) -> np.ndarray:
    """Predict scaled targets for X -> (N, 24, 7)."""
    preds = [models[var].predict(X) for var in C.TARGETS]      # each (N, 24)
    return np.stack(preds, axis=-1)                            # (N, 24, 7)


def save_models(models, history, seed: int) -> None:
    C.ensure_dirs()
    for var, model in models.items():
        model.save_model(str(C.MODELS_DIR / f"xgb_seed{seed}_{var}.json"))
    with open(C.MODELS_DIR / f"history_seed{seed}.json", "w") as f:
        json.dump(history, f, indent=2)
