from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from src.config import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
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
from src.evaluate import (  # noqa: E402
    bootstrap_overall_ci,
    compute_metrics,
    rolling_origin_forecast,
    save_forecasts,
    save_metrics,
)
from src.plots import write_plots  # noqa: E402
from src.sarima_model import fit_sarima_models, parse_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the protocol-compatible SARIMA weather baseline."
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
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--order",
        default=",".join(map(str, DEFAULT_ORDER)),
        help="Non-seasonal ARIMA order as p,d,q.",
    )
    parser.add_argument(
        "--seasonal-order",
        default=",".join(map(str, DEFAULT_SEASONAL_ORDER)),
        help="Seasonal ARIMA order as P,D,Q,s.",
    )
    parser.add_argument(
        "--trend",
        default=DEFAULT_TREND,
        choices=["n", "c", "t", "ct"],
        help="SARIMAX trend term.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon in hours.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Rolling-origin step in hours.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help="Maximum optimizer iterations per target model.",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Limit rolling origins for a quick development run.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap samples for overall MAE/RMSE CIs. Use 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for bootstrap resampling only. SARIMA fitting is deterministic.",
    )
    parser.add_argument(
        "--fit-through-val",
        action="store_true",
        help="Fit SARIMA parameters on 2021-2024 instead of train only.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing matplotlib plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate preprocessing and splits without fitting models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    order = parse_order(args.order, expected_len=3, label="order")
    seasonal_order = parse_order(
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
    print(f"Forecast shape per origin: ({args.horizon}, {len(TARGET_COLUMNS)})")

    if args.dry_run:
        print("Dry run complete. No SARIMA models were fit.")
        return 0

    args.results.mkdir(parents=True, exist_ok=True)

    train_targets = splits.train[TARGET_COLUMNS]
    val_targets = splits.val[TARGET_COLUMNS]
    test_targets = splits.test[TARGET_COLUMNS]

    fit_targets = (
        pd.concat([train_targets, val_targets])
        if args.fit_through_val
        else train_targets
    )
    pretest_update = None if args.fit_through_val else val_targets
    fit_scope = "2021-2024" if args.fit_through_val else "2021-2023"

    print(
        "Fitting seven univariate SARIMA models:",
        f"order={order}",
        f"seasonal_order={seasonal_order}",
        f"trend={args.trend}",
        f"fit_scope={fit_scope}",
    )

    adf_diagnostics = run_pre_fit_diagnostics(train_targets)
    models = fit_sarima_models(
        fit_targets,
        order=order,
        seasonal_order=seasonal_order,
        trend=args.trend,
        maxiter=args.maxiter,
    )
    model_diagnostics = run_post_fit_diagnostics(models)

    diagnostics_text = format_diagnostics(
        adf_diagnostics=adf_diagnostics,
        model_diagnostics=model_diagnostics,
        order=order,
        seasonal_order=seasonal_order,
        trend=args.trend,
        fit_scope=fit_scope,
    )
    (args.results / "diagnostics.txt").write_text(diagnostics_text, encoding="utf-8")

    print("Evaluating rolling-origin forecasts over 2025...")
    forecast_result = rolling_origin_forecast(
        models=models,
        pretest_update=pretest_update,
        test_targets=test_targets,
        horizon=args.horizon,
        step=args.step,
        max_origins=args.max_origins,
    )
    metrics = compute_metrics(forecast_result, variables=TARGET_COLUMNS)

    if args.bootstrap > 0:
        ci = bootstrap_overall_ci(
            forecast_result,
            samples=args.bootstrap,
            seed=args.seed,
        )
        for key, value in ci.items():
            metrics.loc[metrics["scope"] == "overall", key] = value

    save_metrics(metrics, args.results)
    save_forecasts(forecast_result, TARGET_COLUMNS, args.results / "forecasts.csv")

    if not args.no_plots:
        write_plots(metrics, forecast_result, TARGET_COLUMNS, args.results / "plots")

    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print(
        "Done.",
        f"overall MAE={overall['mae']:.4f}",
        f"overall RMSE={overall['rmse']:.4f}",
        f"origins={len(forecast_result.origins):,}",
    )
    print("Results written to:", args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
