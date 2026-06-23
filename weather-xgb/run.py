"""End-to-end XGBoost weather-forecasting pipeline.

Run from the project root:

    python weather-xgb/run.py            # full: 3 seeds, dense train windows
    python weather-xgb/run.py --smoke    # fast smoke test (1 seed, small model)

Writes weather-xgb/results/{metrics.csv, diagnostics.txt, plots/}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `src` importable

from src import config as C
from src import data_prep, diagnostics, evaluate, xgb_model


def _apply_smoke():
    """Shrink the run so the full pipeline executes in ~1 minute end-to-end."""
    C.SEEDS = [0]
    C.TRAIN_STEP = 12
    C.XGB_PARAMS["n_estimators"] = 120
    C.EARLY_STOPPING_ROUNDS = 15


def _validate(df):
    assert df.index.is_monotonic_increasing, "index must be sorted ascending"
    assert df.index.freqstr in ("h", "H"), f"expected hourly freq, got {df.index.freqstr}"
    assert int(df.isna().sum().sum()) == 0, "no nulls allowed after preprocessing"
    print(f"[data] rows={len(df)}  span={df.index[0]} -> {df.index[-1]}  cols={list(df.columns)}")


def main(smoke: bool = False):
    t_start = time.time()
    if smoke:
        _apply_smoke()
        print("[mode] SMOKE — reduced seeds/estimators/train-density")
    C.ensure_dirs()

    # ---- data ----
    df = data_prep.load_clean()
    _validate(df)
    ds = data_prep.build_dataset(df)

    (tr_o, (Xtr, Ytr_s, _)), (va_o, (Xva, Yva_s, Yva_r)), (te_o, (Xte, _, Yte_r)) = \
        data_prep.split_windows(ds)
    print(f"[windows] train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}  "
          f"(input dim={Xtr.shape[1]}, targets={Ytr_s.shape[1:]})")
    print(f"[origins] test={len(te_o)} rolling 24 h windows over "
          f"{ds.index[te_o[0]+1].date()} .. {ds.index[te_o[-1]+C.HORIZON].date()}")

    # ---- train + diagnose + predict, per seed ----
    preds_per_seed, seed_overall = [], []
    for seed in C.SEEDS:
        print(f"\n[seed {seed}] training 7 multi-output regressors ...")
        models, history = xgb_model.train_models(Xtr, Ytr_s, Xva, Yva_s, seed)
        xgb_model.save_models(models, history, seed)
        diag = diagnostics.run_diagnostics(ds, models, history, Xva, Yva_r, seed)
        diagnostics.write_diagnostics_txt(diag, seed)

        preds = evaluate.predict_test(ds, models, Xte)
        preds_per_seed.append(preds)
        m = evaluate.compute_metrics(preds, Yte_r)
        seed_overall.append((m["overall_mae"], m["overall_rmse"]))
        print(f"[seed {seed}] test overall  MAE={m['overall_mae']:.4f}  "
              f"RMSE={m['overall_rmse']:.4f}")

    # ---- ensemble across seeds ----
    preds_ens = np.mean(preds_per_seed, axis=0)
    metrics = evaluate.compute_metrics(preds_ens, Yte_r)
    boot = evaluate.bootstrap_overall(preds_ens, Yte_r)

    seed_overall = np.array(seed_overall)
    seed_summary = dict(
        n_seeds=len(C.SEEDS),
        mae_std=float(seed_overall[:, 0].std()),
        rmse_std=float(seed_overall[:, 1].std()),
    )

    evaluate.write_metrics_csv(metrics, boot, seed_summary)
    evaluate.plot_per_horizon(metrics)
    evaluate.plot_per_variable(metrics)
    evaluate.plot_pred_vs_actual(ds, te_o, preds_ens, Yte_r)

    # Per-origin forecasts for paired significance testing (DM/MCS).
    # origin_time = calendar timestamp of horizon-1 for each test window.
    origin_time = np.array(ds.index[np.asarray(te_o) + 1].values, dtype="datetime64[ns]")
    np.savez(C.RESULTS_DIR / "forecasts_eval.npz",
             y_true=Yte_r, y_pred=preds_ens,
             targets=np.array(list(C.TARGETS)), origin_time=origin_time)

    # ---- summary ----
    print("\n" + "=" * 60)
    print("FINAL — 2025 test, real scale (ensemble of seeds " + str(C.SEEDS) + ")")
    print("=" * 60)
    print(f"Overall MAE  = {metrics['overall_mae']:.4f}  "
          f"95% CI [{boot['mae_lo']:.4f}, {boot['mae_hi']:.4f}]  "
          f"(seed std {seed_summary['mae_std']:.4f})")
    print(f"Overall RMSE = {metrics['overall_rmse']:.4f}  "
          f"95% CI [{boot['rmse_lo']:.4f}, {boot['rmse_hi']:.4f}]  "
          f"(seed std {seed_summary['rmse_std']:.4f})")
    print("\nPer-variable MAE / RMSE:")
    for v, var in enumerate(C.TARGETS):
        print(f"  {var:<9} MAE={metrics['var_mae'][v]:.4f}  "
              f"RMSE={metrics['var_rmse'][v]:.4f}")
    print(f"\nArtifacts: {C.METRICS_CSV}")
    print(f"           {C.DIAGNOSTICS_TXT}")
    print(f"           {C.PLOTS_DIR}/")
    print(f"[done] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast reduced run")
    args = ap.parse_args()
    main(smoke=args.smoke)
