"""
Evaluate DLinear on test set — theo PROTOCOL.md §6-8.

Metrics computed on DENORMALIZED (real-scale) values.
Reports: overall, per-horizon, per-variable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_processed_data
from models.dlinear import DLinear

import json
import torch, torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_prediction_vs_actual


# ============================================================
# Metrics (on real scale, per PROTOCOL.md §6)
# ============================================================

def mae(y_t, y_p):
    return np.mean(np.abs(y_t - y_p))

def rmse(y_t, y_p):
    return np.sqrt(np.mean((y_t - y_p) ** 2))

def mape(y_t, y_p):
    return np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100


def compute_metrics_per_variable(Y_true, Y_pred, target_names):
    """Per-variable MAE, RMSE, MAPE (real scale)."""
    N, H, C = Y_true.shape
    results = {}
    for i, name in enumerate(target_names):
        yt = Y_true[:, :, i].ravel()
        yp = Y_pred[:, :, i].ravel()
        results[name] = {
            'MAE': mae(yt, yp),
            'RMSE': rmse(yt, yp),
            'MAPE': mape(yt, yp),
        }
    return results


def compute_metrics_per_horizon(Y_true, Y_pred, target_names):
    """Per-horizon MAE, RMSE (real scale) — PROTOCOL §6, breakdown 2."""
    N, H, C = Y_true.shape
    horizons = {}
    for h in range(H):
        yt = Y_true[:, h, :].ravel()
        yp = Y_pred[:, h, :].ravel()
        horizons[h + 1] = {'MAE': mae(yt, yp), 'RMSE': rmse(yt, yp)}
    return horizons


def compute_overall(Y_true, Y_pred):
    """Single MAE & RMSE averaged over all horizons, variables, origins."""
    yt = Y_true.ravel()
    yp = Y_pred.ravel()
    return {'MAE': mae(yt, yp), 'RMSE': rmse(yt, yp)}


def bootstrap_ci(Y_true, Y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """Bootstrap CI on a metric (PROTOCOL §6 optional)."""
    N = Y_true.shape[0]
    values = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        idx = rng.randint(0, N, N)
        values[i] = metric_fn(Y_true[idx].ravel(), Y_pred[idx].ravel())
    alpha = (100 - ci) / 2
    lo = np.percentile(values, alpha)
    hi = np.percentile(values, 100 - alpha)
    return lo, hi


def naive_baseline_denorm(Y_true, Y_pred):
    """Naive repeat-last-value baseline (real scale)."""
    N, H, C = Y_true.shape
    # Repeat last known value H times (done during prediction already)
    # Here Y_pred is the naive prediction on real scale
    return {'MAE': mae(Y_true.ravel(), Y_pred.ravel()),
            'RMSE': rmse(Y_true.ravel(), Y_pred.ravel())}


# ============================================================
# Core evaluation (one checkpoint → metrics)
# ============================================================

def _evaluate_checkpoint(cp_path, test_loader, scaler, n_input, n_in, n_out, target_names):
    """Evaluate one checkpoint and return metrics + predictions."""
    checkpoint = torch.load(cp_path, map_location=config.DEVICE, weights_only=False)
    model_type = checkpoint.get('model_type', 'DLinear-S')

    model = DLinear(config.LOOKBACK, config.FORECAST, n_in, out_features=n_out,
                    kernel_size=config.KERNEL_SIZE,
                    individual=config.is_individual(model_type)).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ROLLING_STEP = 24
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(config.DEVICE)
            all_preds.append(model(X).cpu().numpy())
            all_trues.append(Y.cpu().numpy())

    Yp_all = np.concatenate(all_preds, axis=0)
    Yt_all = np.concatenate(all_trues, axis=0)
    Yp_scaled = Yp_all[::ROLLING_STEP]
    Yt_scaled = Yt_all[::ROLLING_STEP]

    target_mean = scaler.mean_[n_input:]
    target_scale = scaler.scale_[n_input:]
    Yp_real = Yp_scaled * target_scale + target_mean
    Yt_real = Yt_scaled * target_scale + target_mean

    overall = compute_overall(Yt_real, Yp_real)
    per_var = compute_metrics_per_variable(Yt_real, Yp_real, target_names)
    per_horizon = compute_metrics_per_horizon(Yt_real, Yp_real, target_names)

    return {
        'model_type': model_type,
        'overall': overall,
        'per_var': per_var,
        'per_horizon': per_horizon,
        'Yp_real': Yp_real,
        'Yt_real': Yt_real,
        'checkpoint': checkpoint,
    }


def _discover_checkpoints():
    """Find all available checkpoints. Returns list of (model_type, seed_label, path)."""
    import glob as _glob
    checkpoints = []

    for model_type, ckpt_dir in [("DLinear-S", config.CHECKPOINT_DIR),
                                  ("DLinear-I", config.CHECKPOINT_DIR_I)]:
        # Multi-seed checkpoints
        multi = sorted(_glob.glob(os.path.join(ckpt_dir, 'best_model_seed*.pth')))
        for cp in multi:
            base = os.path.basename(cp)
            seed_str = base.replace('best_model_seed', '').replace('.pth', '')
            checkpoints.append((model_type, f'seed_{seed_str}', cp))

        # Single-seed fallback
        single = os.path.join(ckpt_dir, 'best_model.pth')
        if not multi and os.path.exists(single):
            checkpoints.append((model_type, 'single', single))

    return checkpoints


# ============================================================
# Evaluation (single or multi-seed)
# ============================================================

def evaluate():
    print("=" * 70)
    print(" DLinear Weather Forecasting — EVALUATION (PROTOCOL §6-8)")
    print("=" * 70)

    checkpoints = _discover_checkpoints()
    if not checkpoints:
        print("No checkpoint found! Run train.py first."); return

    # -- Data -----------------------------------------------
    train_loader, val_loader, test_loader, scaler, _, _ = load_processed_data()
    # Load first checkpoint to get feature/target names
    first_path = checkpoints[0][2]
    cp0 = torch.load(first_path, map_location=config.DEVICE, weights_only=False)
    feature_names = cp0['feature_names']
    target_names = cp0.get('target_names', feature_names)
    n_in = len(feature_names)
    n_out = len(target_names)
    n_input = n_in

    # -- Evaluate each checkpoint ---------------------------
    results = []
    for model_type, label, cp_path in checkpoints:
        print(f"\n{'#'*60}\n# {model_type} {label}: {os.path.basename(cp_path)}\n{'#'*60}")
        r = _evaluate_checkpoint(cp_path, test_loader, scaler, n_input, n_in, n_out, target_names)
        results.append((model_type, label, r))

    # -- Multi-seed summary grouped by model type ------------
    for mt in ("DLinear-S", "DLinear-I"):
        mt_results = [(l, r) for m, l, r in results if m == mt]
        if len(mt_results) > 1:
            print(f"\n{'='*60}")
            print(f" {mt} MULTI-SEED SUMMARY (n={len(mt_results)})")
            print(f"{'='*60}")
            maes = [r['overall']['MAE'] for _, r in mt_results]
            rmses = [r['overall']['RMSE'] for _, r in mt_results]
            for label, r in mt_results:
                print(f"  {label}: MAE={r['overall']['MAE']:.4f}  RMSE={r['overall']['RMSE']:.4f}")
            print(f"  {'Mean':<12} MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}")
            print(f"  {'':<12} RMSE={np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # -- S vs I comparison (use mean of multi-seed or single) --
    def _agg(model_type):
        mt = [r for m, _, r in results if m == model_type]
        if not mt:
            return None
        maes = [r['overall']['MAE'] for r in mt]
        rmses = [r['overall']['RMSE'] for r in mt]
        return {'MAE': np.mean(maes), 'MAE_std': np.std(maes),
                'RMSE': np.mean(rmses), 'RMSE_std': np.std(rmses),
                'per_var': mt[0]['per_var']}

    s_res = _agg("DLinear-S")
    i_res = _agg("DLinear-I")

    if s_res and i_res:
        impr_mae = (s_res['MAE'] - i_res['MAE']) / s_res['MAE'] * 100
        impr_rmse = (s_res['RMSE'] - i_res['RMSE']) / s_res['RMSE'] * 100
        print(f"\n{'='*60}")
        print(f" DLinear-S vs DLinear-I COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Metric':<15} {'DLinear-S':<18} {'DLinear-I':<18} {'Improvement':<12}")
        print(f"  {'-'*57}")
        print(f"  {'MAE':<15} {s_res['MAE']:.4f} ± {s_res['MAE_std']:.4f}  {i_res['MAE']:.4f} ± {i_res['MAE_std']:.4f}  {impr_mae:+.1f}%")
        print(f"  {'RMSE':<15} {s_res['RMSE']:.4f} ± {s_res['RMSE_std']:.4f}  {i_res['RMSE']:.4f} ± {i_res['RMSE_std']:.4f}  {impr_rmse:+.1f}%")
        print()
        print(f"  {'Variable':<30} {'S MAE':<10} {'I MAE':<10} {'Delta%':<10}")
        print(f"  {'-'*60}")
        for name in target_names:
            s_mae = s_res['per_var'][name]['MAE']
            i_mae = i_res['per_var'][name]['MAE']
            delta = (s_mae - i_mae) / s_mae * 100
            print(f"  {name:<30} {s_mae:<10.4f} {i_mae:<10.4f} {delta:+.1f}%")

    # -- Use first DLinear-S result for detailed plots ----
    s_first = [(l, r) for m, l, r in results if m == "DLinear-S"]
    if not s_first:
        s_first = [(l, r) for m, l, r in results[:1]]
    label, r0 = s_first[0]
    Yp_real = r0['Yp_real']
    Yt_real = r0['Yt_real']
    overall = r0['overall']
    per_var = r0['per_var']
    per_horizon = r0['per_horizon']
    N, H, C_out = Yp_real.shape
    print(f"\nRolling-origin: {N} origins x {H}h forecasts (step=24h)")

    # -- Per-model save (S and I separately) --
    for mt in ("DLinear-S", "DLinear-I"):
        mt_list = [(l, r) for m, l, r in results if m == mt]
        if not mt_list:
            continue
        _, r_mt = mt_list[0]
        od = config.get_output_dir(mt)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame(r_mt['per_var']).T.to_csv(os.path.join(od, 'test_metrics.csv'))
        with open(os.path.join(od, 'evaluation_results.json'), 'w') as f:
            json.dump({'model_type': mt, 'overall': r_mt['overall'],
                       'per_var': r_mt['per_var'], 'per_horizon': r_mt['per_horizon']}, f, indent=2)
        print(f"[{mt}] Results saved to {od}/")

    # ========================================================
    # Report 1: Overall (PROTOCOL §6 breakdown 1)
    # ========================================================
    print(f"\n{'='*60}")
    print(f" OVERALL (real scale)")
    print(f"{'='*60}")
    print(f"  MAE  = {overall['MAE']:.4f}")
    print(f"  RMSE = {overall['RMSE']:.4f}")

    mae_lo, mae_hi = bootstrap_ci(Yt_real, Yp_real, mae)
    rmse_lo, rmse_hi = bootstrap_ci(Yt_real, Yp_real, rmse)
    print(f"  MAE  95% CI: [{mae_lo:.4f}, {mae_hi:.4f}]")
    print(f"  RMSE 95% CI: [{rmse_lo:.4f}, {rmse_hi:.4f}]")

    # ========================================================
    # Report 2: Per-variable (PROTOCOL §6 breakdown 3)
    # ========================================================
    print(f"\n{'='*80}")
    print(f"{'Variable':<38}{'MAE':<12}{'RMSE':<12}{'MAPE':<12}")
    print(f"{'='*80}")
    for name in target_names:
        m = per_var[name]
        print(f"{name:<38}{m['MAE']:<12.4f}{m['RMSE']:<12.4f}{m['MAPE']:<12.2f}")
    avg_mae = np.mean([r['MAE'] for r in per_var.values()])
    avg_rmse = np.mean([r['RMSE'] for r in per_var.values()])
    print(f"{'='*80}")
    print(f"{'AVERAGE':<38}{avg_mae:<12.4f}{avg_rmse:<12.4f}{'--':<12}")

    # ========================================================
    # Report 3: Per-horizon (PROTOCOL §6 breakdown 2)
    # ========================================================
    horizon_mae = [per_horizon[h]['MAE'] for h in range(1, H + 1)]
    horizon_rmse = [per_horizon[h]['RMSE'] for h in range(1, H + 1)]
    print(f"\n{'='*60}")
    print(f" PER-HORIZON ERROR")
    print(f"{'='*60}")
    print(f"  Hour  1: MAE={horizon_mae[0]:.4f}  RMSE={horizon_rmse[0]:.4f}  (closest)")
    print(f"  Hour 12: MAE={horizon_mae[11]:.4f}  RMSE={horizon_rmse[11]:.4f}  (mid)")
    print(f"  Hour 24: MAE={horizon_mae[-1]:.4f}  RMSE={horizon_rmse[-1]:.4f}  (farthest)")

    # ========================================================
    # Plot: Per-horizon error curve (PROTOCOL §7 required)
    # ========================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    hours = np.arange(1, H + 1)
    ax1.plot(hours, horizon_mae, 'b-', lw=2, label='MAE')
    ax1.fill_between(hours, 0, horizon_mae, alpha=0.1, color='blue')
    ax1.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax1.set_ylabel('MAE (real scale)', fontweight='bold')
    ax1.set_title('Per-Horizon MAE — Error Accumulation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(hours, horizon_rmse, 'r-', lw=2, label='RMSE')
    ax2.fill_between(hours, 0, horizon_rmse, alpha=0.1, color='red')
    ax2.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax2.set_ylabel('RMSE (real scale)', fontweight='bold')
    ax2.set_title('Per-Horizon RMSE — Error Accumulation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    print(f"  test_metrics.csv, evaluation_results.json, per_horizon_error.png")
    print(f"  pred_vs_actual/ (per-variable predicted-vs-actual plots)")

    return Yp_real, Yt_real, per_var, target_names, scaler


if __name__ == "__main__":
    evaluate()
