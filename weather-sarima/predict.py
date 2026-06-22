from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from src.config import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_HORIZON,
    DEFAULT_RESULTS_DIR,
    DEFAULT_STEP,
    TARGET_COLUMNS,
)
from src.data_prep import load_weather_data, split_by_protocol  # noqa: E402
from src.evaluate import (  # noqa: E402
    bootstrap_overall_ci,
    compute_metrics,
    rolling_origin_forecast,
    save_forecasts,
    save_metrics,
)
from src.plots import write_plots  # noqa: E402
from src.sarima_model import load_model_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load saved SARIMA models and run rolling-origin prediction."
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
        help="Directory containing models and where predictions are written.",
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
        help="Seed for bootstrap resampling only.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing matplotlib plots.",
    )
    return parser.parse_args()


def infer_fit_through_val(
    metadata: dict[str, Any],
    best_models: pd.DataFrame,
    train_rows: int,
    val_rows: int,
) -> bool:
    fit_through_val = metadata.get("fit_through_val")
    if isinstance(fit_through_val, bool):
        return fit_through_val

    if "final_fit_rows" in best_models.columns:
        final_rows = pd.to_numeric(
            best_models["final_fit_rows"],
            errors="coerce",
        ).dropna()
        unique_final_rows = final_rows.unique()
        if len(unique_final_rows) == 1:
            final_fit_rows = int(unique_final_rows[0])
            if final_fit_rows == train_rows + val_rows:
                return True
            if final_fit_rows == train_rows:
                return False

    return False


def main() -> int:
    args = parse_args()

    data = load_weather_data(args.data)
    splits = split_by_protocol(data)
    test_targets = splits.test[TARGET_COLUMNS]
    val_targets = splits.val[TARGET_COLUMNS]

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

    try:
        models, best_models, metadata = load_model_artifacts(
            args.results,
            variables=TARGET_COLUMNS,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Prediction setup failed: {exc}", file=sys.stderr)
        return 1
    fit_through_val = infer_fit_through_val(
        metadata=metadata,
        best_models=best_models,
        train_rows=len(splits.train),
        val_rows=len(splits.val),
    )
    pretest_update = None if fit_through_val else val_targets

    print("Loaded saved SARIMA models from:", args.results / "models")
    print(
        "Evaluating rolling-origin forecasts over 2025:",
        f"fit_through_val={fit_through_val}",
        f"horizon={args.horizon}",
        f"step={args.step}",
    )

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

    args.results.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, args.results)
    save_forecasts(forecast_result, TARGET_COLUMNS, args.results / "forecasts.csv")

    if not args.no_plots:
        write_plots(metrics, forecast_result, TARGET_COLUMNS, args.results / "plots")

    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    print(
        "Prediction complete.",
        f"overall MAE={overall['mae']:.4f}",
        f"overall RMSE={overall['rmse']:.4f}",
        f"origins={len(forecast_result.origins):,}",
    )
    print("Prediction outputs written to:", args.results)
    return 0


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        raise SystemExit(main())
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
