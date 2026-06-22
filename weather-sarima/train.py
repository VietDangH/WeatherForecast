from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from src.config import (  # noqa: E402
    DEFAULT_HORIZON,
    DEFAULT_MAXITER,
    DEFAULT_ORDER,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEASONAL_ORDER,
    DEFAULT_STEP,
    DEFAULT_TREND,
    TARGET_COLUMNS,
)
from src.data_prep import load_weather_data, split_by_protocol  # noqa: E402
from src.diagnostics import (  # noqa: E402
    format_diagnostics,
    run_post_fit_diagnostics,
    run_pre_fit_diagnostics,
)
from src.sarima_model import (  # noqa: E402
    fit_best_sarima_models,
    parse_order_grid,
    save_model_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, select, and save SARIMA weather models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "weather_dataset.csv",
        help="Path to the weather CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where model artifacts are written.",
    )
    parser.add_argument(
        "--order",
        default=",".join(map(str, DEFAULT_ORDER)),
        help=(
            "Non-seasonal ARIMA order candidates as p,d,q. "
            "Use semicolons for a grid, e.g. '1,0,1;2,0,1'."
        ),
    )
    parser.add_argument(
        "--seasonal-order",
        default=",".join(map(str, DEFAULT_SEASONAL_ORDER)),
        help=(
            "Seasonal ARIMA order candidates as P,D,Q,s. "
            "Use semicolons for a grid, e.g. '1,0,1,24;0,0,1,24'."
        ),
    )
    parser.add_argument(
        "--trend",
        default=DEFAULT_TREND,
        choices=["n", "c", "t", "ct"],
        help="SARIMAX trend term.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help="Maximum optimizer iterations per target model.",
    )
    parser.add_argument(
        "--selection-horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Validation horizon for selecting the best model.",
    )
    parser.add_argument(
        "--selection-step",
        type=int,
        default=DEFAULT_STEP,
        help="Validation rolling step for selecting the best model.",
    )
    parser.add_argument(
        "--fit-through-val",
        action="store_true",
        help="Fit selected SARIMA parameters on 2021-2024 instead of train only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate preprocessing and splits without fitting models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    order_candidates = parse_order_grid(args.order, expected_len=3, label="order")
    seasonal_order_candidates = parse_order_grid(
        args.seasonal_order, expected_len=4, label="seasonal-order"
    )

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

    if args.dry_run:
        print("Dry run complete. No SARIMA models were fit.")
        return 0

    args.results.mkdir(parents=True, exist_ok=True)

    train_targets = splits.train[TARGET_COLUMNS]
    val_targets = splits.val[TARGET_COLUMNS]
    fit_targets = (
        pd.concat([train_targets, val_targets])
        if args.fit_through_val
        else train_targets
    )
    fit_scope = "2021-2024" if args.fit_through_val else "2021-2023"

    print(
        "Selecting and fitting seven univariate SARIMA models:",
        f"order_candidates={order_candidates}",
        f"seasonal_order_candidates={seasonal_order_candidates}",
        f"trend={args.trend}",
        f"fit_scope={fit_scope}",
        f"selection_horizon={args.selection_horizon}",
        f"selection_step={args.selection_step}",
    )

    adf_diagnostics = run_pre_fit_diagnostics(train_targets)
    models, best_models, model_selection = fit_best_sarima_models(
        train_targets=train_targets,
        validation_targets=val_targets,
        final_fit_targets=fit_targets,
        order_candidates=order_candidates,
        seasonal_order_candidates=seasonal_order_candidates,
        trend=args.trend,
        maxiter=args.maxiter,
        selection_horizon=args.selection_horizon,
        selection_step=args.selection_step,
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
        "order_candidates": [list(order) for order in order_candidates],
        "seasonal_order_candidates": [
            list(seasonal_order) for seasonal_order in seasonal_order_candidates
        ],
        "trend": args.trend,
        "selection_horizon": args.selection_horizon,
        "selection_step": args.selection_step,
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
        order_candidates=order_candidates,
        seasonal_order_candidates=seasonal_order_candidates,
        trend=args.trend,
        fit_scope=fit_scope,
        best_models=best_models,
    )
    (args.results / "diagnostics.txt").write_text(diagnostics_text, encoding="utf-8")

    print("Training complete. Model artifacts written to:", args.results / "models")
    return 0


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        raise SystemExit(main())
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
