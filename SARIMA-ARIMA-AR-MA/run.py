"""End-to-end per-feature ARIMA-family weather baseline.

Pipeline:
  1. Load + split data (shared PROTOCOL).
  2. Pre-fit ADF diagnostics on train.
  3. PER-FEATURE selection: each target searches its own grid
     (config.PER_FEATURE_GRID), best chosen by rolling-origin validation MAE.
  4. Post-fit diagnostics (Ljung-Box, Jarque-Bera) on the selected models.
  5. Rolling-origin evaluation over the 2025 test year.
  6. Write metrics, forecasts, diagnostics, plots, and saved models.

Usage:
  python SARIMA-ARIMA-AR-MA/run.py                 # full run
  python SARIMA-ARIMA-AR-MA/run.py --jobs 3        # parallel across features
  python SARIMA-ARIMA-AR-MA/run.py --max-origins 5 # quick smoke test
  python SARIMA-ARIMA-AR-MA/run.py --dry-run       # check data/splits only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from config import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_DATA_PATH,
    DEFAULT_HORIZON,
    DEFAULT_MAXITER,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEED,
    DEFAULT_STEP,
    PER_FEATURE_GRID,
    TARGET_COLUMNS,
)
from data_prep import load_weather_data, split_by_protocol  # noqa: E402
from diagnostics import (  # noqa: E402
    format_diagnostics,
    run_post_fit_diagnostics,
    run_pre_fit_diagnostics,
)
from evaluate import (  # noqa: E402
    bootstrap_overall_ci,
    compute_metrics,
    rolling_origin_forecast,
    save_forecasts,
    save_metrics,
)
from models import make_model_specs, save_model_artifacts, select_all  # noqa: E402
from plots import write_plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument(
        "--selection-horizon", type=int, default=None,
        help="Validation horizon for model selection (defaults to --horizon).",
    )
    parser.add_argument(
        "--selection-step", type=int, default=None,
        help="Validation step for model selection (defaults to --step).",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel workers across the 7 features (needs joblib). "
             "Each worker holds a full SARIMAX fit in memory.",
    )
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fit-through-val", action="store_true",
        help="Refit selected models on 2021-2024 before testing.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_horizon = args.selection_horizon or args.horizon
    selection_step = args.selection_step or args.step

    data = load_weather_data(args.data)
    splits = split_by_protocol(data)

    print("Loaded dataset:", args.data)
    print(f"Rows: {len(data):,} | columns: {len(data.columns)}")
    print(
        "Split rows:",
        f"train={len(splits.train):,}",
        f"val={len(splits.val):,}",
        f"test={len(splits.test):,}",
    )
    print("Targets:", ", ".join(TARGET_COLUMNS))
    print(f"Forecast shape per origin: ({args.horizon}, {len(TARGET_COLUMNS)})")
    print("Per-feature candidate counts:",
          {v: len(PER_FEATURE_GRID[v]) for v in TARGET_COLUMNS})

    if args.dry_run:
        print("Dry run complete. No models were fit.")
        return 0

    args.results.mkdir(parents=True, exist_ok=True)

    train_targets = splits.train[TARGET_COLUMNS]
    val_targets = splits.val[TARGET_COLUMNS]
    test_targets = splits.test[TARGET_COLUMNS]

    fit_targets = (
        pd.concat([train_targets, val_targets]) if args.fit_through_val else train_targets
    )
    pretest_update = None if args.fit_through_val else val_targets
    fit_scope = "2021-2024" if args.fit_through_val else "2021-2023"

    print(f"\nPer-feature selection (fit_scope={fit_scope}, jobs={args.jobs})...")
    adf_diagnostics = run_pre_fit_diagnostics(train_targets)
    models, best_models, model_selection = select_all(
        train_targets=train_targets,
        validation_targets=val_targets,
        final_fit_targets=fit_targets,
        grid=PER_FEATURE_GRID,
        maxiter=args.maxiter,
        selection_horizon=selection_horizon,
        selection_step=selection_step,
        n_jobs=args.jobs,
    )
    model_diagnostics = run_post_fit_diagnostics(models)

    metadata = {
        "data_path": str(args.data),
        "target_columns": TARGET_COLUMNS,
        "fit_through_val": bool(args.fit_through_val),
        "fit_scope": fit_scope,
        "train_rows": len(train_targets),
        "validation_rows": len(val_targets),
        "final_fit_rows": len(fit_targets),
        "per_feature_grid": {v: [list(map(list, c)) for c in PER_FEATURE_GRID[v]]
                             for v in TARGET_COLUMNS},
        "selection_horizon": selection_horizon,
        "selection_step": selection_step,
    }
    save_model_artifacts(
        models=models,
        best_models=best_models,
        selection_results=model_selection,
        results_dir=args.results,
        metadata=metadata,
    )

    diagnostics_text = format_diagnostics(
        adf_diagnostics=adf_diagnostics,
        model_diagnostics=model_diagnostics,
        best_models=best_models,
        fit_scope=fit_scope,
    )
    (args.results / "diagnostics.txt").write_text(diagnostics_text, encoding="utf-8")

    print("\nRolling-origin evaluation over 2025...")
    model_specs = make_model_specs(models)
    forecast_result = rolling_origin_forecast(
        model_specs=model_specs,
        pretest_update=pretest_update,
        test_targets=test_targets,
        horizon=args.horizon,
        step=args.step,
        max_origins=args.max_origins,
    )
    metrics = compute_metrics(forecast_result, variables=TARGET_COLUMNS)

    if args.bootstrap > 0:
        ci = bootstrap_overall_ci(forecast_result, samples=args.bootstrap, seed=args.seed)
        for key, value in ci.items():
            metrics.loc[metrics["scope"] == "overall", key] = value

    save_metrics(metrics, args.results)
    save_forecasts(forecast_result, TARGET_COLUMNS, args.results / "forecasts.csv")

    if not args.no_plots:
        write_plots(
            metrics, forecast_result, TARGET_COLUMNS, args.results / "plots", best_models
        )

    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print("\nSelected models:")
    print(best_models[["variable", "family", "order", "seasonal_order",
                       "validation_mae"]].to_string(index=False))
    print(
        "\nDone.",
        f"overall MAE={overall['mae']:.4f}",
        f"overall RMSE={overall['rmse']:.4f}",
        f"origins={len(forecast_result.origins):,}",
    )
    print("Results written to:", args.results)
    return 0


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        raise SystemExit(main())
    finally:
        print(f"Elapsed time: {time.perf_counter() - start:.2f} seconds")
