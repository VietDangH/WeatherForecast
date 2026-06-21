"""
Diagnostics — PROTOCOL.md §6.
Run sau evaluate để validate model.

Methods: Ljung-Box residual autocorrelation, residual histogram/Q-Q plot,
         stability (loss convergence).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_processed_data
from models.dlinear import DLinear

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def _ljung_box(residuals, lag):
    """
    Ljung-Box from definition:
      Q = n*(n+2) * Σ(k=1..lag) ρ̂_k²/(n-k)
    Under H0 (white noise), Q ~ χ²(lag).
    """
    n = len(residuals)
    if n <= lag:
        return np.nan, np.nan

    r = residuals - residuals.mean()
    acf = np.correlate(r, r, mode='full')[n - 1:]
    acf = acf / acf[0]

    q_stat = 0.0
    for k in range(1, lag + 1):
        if k < len(acf):
            q_stat += (acf[k] ** 2) / (n - k)
    q_stat *= n * (n + 2)

    p_val = 1 - stats.chi2.cdf(q_stat, lag)
    return q_stat, p_val


def compute_residuals(model, test_loader, scaler, n_in, n_out, step=24):
    """Residuals (actual - predicted) on real scale — subsampled by step for non-overlap."""
    model.eval()
    residuals = []
    t_mean = scaler.mean_[n_in:]
    t_scale = scaler.scale_[n_in:]

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(config.DEVICE)
            Yp = model(X).cpu().numpy()
            Yt = Y.cpu().numpy()
            residuals.append(Yt - Yp)  # residuals on scaled scale

    res_all = np.concatenate(residuals, axis=0)
    # Subsample with same step as rolling-origin evaluation
    res_sampled = res_all[::step]
    # Denormalize: (ŷ_s - y_s) * σ = ŷ_real - y_real  (mean cancels out)
    return res_sampled * t_scale


def ljung_box_test(residuals, target_names, lag=24):
    """
    PROTOCOL §6: Ljung-Box — H0: white noise residuals.

    Tests residuals at each forecast horizon separately so the result
    has diagnostic value (pooling 200K+ residuals makes any model fail).

    Reports: pass rate per variable, worst horizon, overall verdict.
    p > 0.05 = 'pass' (no significant residual autocorrelation).
    """
    N, H, C = residuals.shape

    print(f"\n{'='*70}")
    print(f" LJUNG-BOX TEST (lag={lag}) — per-horizon residual autocorrelation")
    print(f" H0: white noise residuals. p>0.05 = PASS.")
    print(f" ({N} origins × {H} horizons)")
    print(f"{'='*70}")
    print(f"{'Variable':<30}{'Worst h':>8}{'Q-stat':>10}{'p-value':>10}{'Passed':>8}")
    print(f"{'-'*70}")

    results = {}
    all_good = True
    for i, name in enumerate(target_names):
        n_pass = 0
        worst_h, worst_stat, worst_p = None, -np.inf, 0
        for h in range(H):
            resid_h = residuals[:, h, i]  # shape (N_origins,)
            stat_val, p_val = _ljung_box(resid_h, lag)
            if not np.isnan(p_val) and p_val > 0.05:
                n_pass += 1
            if stat_val > worst_stat:
                worst_h, worst_stat, worst_p = h + 1, stat_val, p_val

        all_pass = n_pass == H
        if not all_pass:
            all_good = False
        results[name] = {
            'pass_rate': f"{n_pass}/{H}",
            'worst_horizon': worst_h,
            'worst_statistic': float(worst_stat),
            'worst_p_value': float(worst_p) if not np.isnan(worst_p) else 0.0,
        }
        print(f"{name:<30}{worst_h:>8}{worst_stat:>10.2f}{worst_p:>10.4f}{n_pass:>5}/{H}")

    if all_good:
        print(f"\nAll residuals white noise across all horizons — model well-specified.")
    else:
        total_pass = sum(int(r['pass_rate'].split('/')[0]) for r in results.values())
        total_tests = C * H
        print(f"\nResidual autocorrelation detected — {total_pass}/{total_tests} horizon-variable pairs pass.")
        print("  Near-horizon residuals (h=1..6) typically show strongest autocorrelation.")
        print("  Consider: increase lookback, use DLinear-I, or add feature engineering.")

    return results


def residual_normality_check(residuals, target_names, save_dir):
    """PROTOCOL §6: Residual histogram + Q-Q plot + D'Agostino-Pearson test."""
    _, _, C = residuals.shape
    n_cols = min(4, C)
    n_rows = (C + n_cols - 1) // n_cols

    # Histogram
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten() if C > 1 else [axes]

    for i, name in enumerate(target_names):
        r = residuals[:, :, i].ravel()
        axes_flat[i].hist(r, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        mu, sigma = r.mean(), r.std()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        axes_flat[i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
        axes_flat[i].set_title(name, fontsize=9)
        axes_flat[i].axvline(0, color='black', ls='--', lw=0.5)

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle('Residual Histograms (Normal overlay)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Residual histograms -> {save_dir}/residual_histograms.png")

    # Q-Q plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes_flat = axes.flatten() if C > 1 else [axes]

    for i, name in enumerate(target_names):
        r = residuals[:, :, i].ravel()
        stats.probplot(r, dist="norm", plot=axes_flat[i])
        axes_flat[i].set_title(name, fontsize=9)

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle('Q-Q Plots (Normal)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_qq_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Q-Q plots -> {save_dir}/residual_qq_plots.png")

    # Normality test
    print(f"\n{'='*70}")
    print(f" NORMALITY TEST (D'Agostino-Pearson) — H0: normal")
    print(f"{'='*70}")
    for i, name in enumerate(target_names):
        r = residuals[:, :, i].ravel()
        s = r if len(r) < 5000 else np.random.RandomState(42).choice(r, 5000)
        st, pv = stats.normaltest(s)
        print(f"  {name:<30} stat={st:8.2f}  p={pv:.4f}  [{'PASS' if pv > 0.05 else 'FAIL'}]")


def forecast_intervals(residuals, target_names, save_dir, ci=95):
    """
    PROTOCOL §6: Forecast intervals — uncertainty bands widening over horizon.

    For deterministic models (DLinear): use empirical residual quantiles
    at each forecast horizon (conformal-style prediction intervals).

    Returns: lower/upper bounds per horizon per variable.
    """
    _, H, C = residuals.shape
    alpha = (100 - ci) / 2
    lo_pct = alpha
    hi_pct = 100 - alpha

    # -- Compute per-horizon intervals --
    intervals = {}
    for i, name in enumerate(target_names):
        intervals[name] = {'lower': np.zeros(H), 'upper': np.zeros(H)}
        for h in range(H):
            resid_h = residuals[:, h, i]
            intervals[name]['lower'][h] = np.percentile(resid_h, lo_pct)
            intervals[name]['upper'][h] = np.percentile(resid_h, hi_pct)

    # -- Plot for key variables (temp, rh, cloud, wx) --
    plot_vars = target_names[:4]  # first 4 targets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    for idx, name in enumerate(plot_vars):
        ax = axes_flat[idx]
        i = target_names.index(name)
        hours = np.arange(1, H + 1)
        lo = intervals[name]['lower']
        hi = intervals[name]['upper']
        mid = (lo + hi) / 2

        ax.fill_between(hours, lo, hi, alpha=0.3, color='steelblue', label=f'{ci}% prediction interval')
        ax.plot(hours, lo, 'b--', lw=1, alpha=0.7)
        ax.plot(hours, hi, 'b--', lw=1, alpha=0.7)
        ax.plot(hours, mid, 'b-', lw=1.5, alpha=0.9, label='Median residual')
        ax.axhline(y=0, color='black', lw=0.8)
        ax.set_xlabel('Forecast Horizon (hours)')
        ax.set_ylabel('Residual (real scale)')
        ax.set_title(f'{name} — Interval widening over horizon', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print widening factor
        spread_t1 = hi[0] - lo[0]
        spread_t24 = hi[-1] - lo[-1]
        ratio = spread_t24 / spread_t1 if spread_t1 > 0 else 1.0
        print(f"  {name:<30} h=1: ±{spread_t1/2:.2f}  h=24: ±{spread_t24/2:.2f}  (×{ratio:.1f})")

    for j in range(len(plot_vars), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(f'Forecast Intervals ({ci}% CI) — Uncertainty Bands', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forecast_intervals.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nForecast intervals -> {save_dir}/forecast_intervals.png")

    return intervals


def adf_stationarity_test(scaler, n_input, target_names):
    """
    PROTOCOL §6: ADF stationarity test (pre-fit) on the FULL train series.

    Uses the processed train data (real scale, denormalized) so the test
    has the correct statistical power and critical values.

    H0: series has a unit root (non-stationary). p<0.05 = stationary.
    """
    from statsmodels.tsa.stattools import adfuller

    data = np.load(config.PROCESSED_DATA_PATH)
    Y_train_scaled = data['Y_train']  # (N_train, 7) — scaled targets

    target_mean = scaler.mean_[n_input:]
    target_scale = scaler.scale_[n_input:]
    Y_train_real = Y_train_scaled * target_scale + target_mean

    print(f"\n{'='*70}")
    print(f" ADF STATIONARITY TEST (on full train, real scale)")
    print(f" H0: unit root (non-stationary). p<0.05 = stationary.")
    print(f"{'='*70}")
    print(f"{'Variable':<30}{'ADF Stat':<14}{'p-value':<14}{'Verdict':<12}")
    print(f"{'-'*70}")

    n_train = Y_train_real.shape[0]
    print(f"  (train series length: {n_train:,} hours)")

    all_stationary = True
    for i, name in enumerate(target_names):
        series = Y_train_real[:, i]
        result = adfuller(series, maxlag=48, autolag='AIC')
        stat_val = result[0]
        p_val = result[1]

        stationary = p_val < 0.05
        if not stationary:
            all_stationary = False
        verdict = 'stationary' if stationary else 'non-stat'
        print(f"{name:<30}{stat_val:<14.4f}{p_val:<14.4f}{verdict:<12}")

    if all_stationary:
        print("\nAll series stationary — d=0 for VAR, no differencing needed.")
    else:
        print("\nSome series non-stationary — VAR would need differencing (d=1).")
        print("  (DL models handle this via learning — no differencing required.)")

    return all_stationary


def bootstrap_ci_metrics(residuals, n_bootstrap=1000):
    """
    PROTOCOL §6: Bootstrap CI on MAE/RMSE.

    Resample forecast windows with replacement → distribution of MAE/RMSE.
    Report 95% CI instead of bare point estimates.
    """
    N = residuals.shape[0]
    abs_err = np.abs(residuals)         # (N, H, C)
    sq_err = residuals ** 2

    rng = np.random.RandomState(42)
    mae_boot = np.zeros(n_bootstrap)
    rmse_boot = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.randint(0, N, N)
        mae_boot[b] = np.mean(abs_err[idx])
        rmse_boot[b] = np.sqrt(np.mean(sq_err[idx]))

    lo_mae, hi_mae = np.percentile(mae_boot, [2.5, 97.5])
    lo_rmse, hi_rmse = np.percentile(rmse_boot, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f" BOOTSTRAP CI ON METRICS (95%, {n_bootstrap} resamples)")
    print(f"{'='*60}")
    print(f"  MAE  = {np.mean(mae_boot):.4f}  [{lo_mae:.4f}, {hi_mae:.4f}]")
    print(f"  RMSE = {np.mean(rmse_boot):.4f}  [{lo_rmse:.4f}, {hi_rmse:.4f}]")

    return {'MAE_ci': (float(lo_mae), float(hi_mae)), 'RMSE_ci': (float(lo_rmse), float(hi_rmse))}


def stability_check(history):
    """PROTOCOL §6: Check val loss not diverging."""
    vl = history.get('val_loss', [])
    if len(vl) >= 10:
        slope, _, _, _, _ = stats.linregress(range(10), np.array(vl[-10:]))
        if slope > 0.001:
            print(f"\nWARNING: Val loss trending UP (slope={slope:.6f}) — instability.")
            return False
        print(f"\nStability OK — val loss flat/trending down (last-10 slope={slope:.6f}).")
        return True
    return True


def diagnostic_verdict(lb_results, all_stationary, is_stable):
    """
    PROTOCOL §6: Summary paragraph — each model must report a diagnostics verdict.
    """
    print(f"\n{'='*70}")
    print(f" DIAGNOSTIC VERDICT")
    print(f"{'='*70}")

    issues = []
    if not all_stationary:
        issues.append("non-stationary series detected (informational for DL)")
    if not all(int(r['pass_rate'].split('/')[0]) == int(r['pass_rate'].split('/')[1]) for r in lb_results.values()):
        issues.append("Ljung-Box: residual autocorrelation -- more lookback/capacity needed")
    if not is_stable:
        issues.append("loss divergence detected")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendation: fix issues before final evaluation.")
    else:
        print("All diagnostics PASS. Model is well-specified and ready for final evaluation.")


def _discover_checkpoints(model_type):
    """Find all checkpoints for a model type. Returns list of (seed_label, path)."""
    import glob as _glob
    ckpt_dir = config.get_checkpoint_dir(model_type)
    checkpoints = []

    # Multi-seed checkpoints
    multi = sorted(_glob.glob(os.path.join(ckpt_dir, 'best_model_seed*.pth')))
    for cp in multi:
        base = os.path.basename(cp)
        seed_str = base.replace('best_model_seed', '').replace('.pth', '')
        checkpoints.append((f'seed_{seed_str}', cp))

    # Fallback to single checkpoint
    if not checkpoints:
        single = os.path.join(ckpt_dir, 'best_model.pth')
        if os.path.exists(single):
            checkpoints.append(('single', single))

    return checkpoints


def run_diagnostics(model_type="DLinear-S"):
    checkpoint_dir = config.get_checkpoint_dir(model_type)

    print("=" * 70)
    print(f" DLinear — DIAGNOSTICS (PROTOCOL §6) [{model_type}]")
    print("=" * 70)

    diag_dir = os.path.join(config.get_output_dir(model_type), 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    # ---- Discover checkpoints (multi-seed) ----
    checkpoints = _discover_checkpoints(model_type)
    if not checkpoints:
        print("No checkpoint! Run train.py first."); return

    print(f"\nFound {len(checkpoints)} checkpoint(s): {[c[0] for c in checkpoints]}")

    # Load first checkpoint for metadata
    cp0 = torch.load(checkpoints[0][1], map_location=config.DEVICE, weights_only=False)
    feature_names = cp0['feature_names']
    target_names = cp0.get('target_names', feature_names)
    history = cp0.get('history', {})
    n_in = len(feature_names)
    n_out = len(target_names)

    _, val_loader, test_loader, scaler, _, _ = load_processed_data()

    # ---- 1. ADF stationarity (pre-fit, on full train) ----
    all_stationary = adf_stationarity_test(scaler, n_in, target_names)

    # ---- Compute predictions from ALL seeds (ensemble average) ----
    print(f"\nComputing ensemble predictions ({len(checkpoints)} seed(s))...")
    all_preds = []
    for label, cp_path in checkpoints:
        model = DLinear(config.LOOKBACK, config.FORECAST, n_in, out_features=n_out,
                        kernel_size=config.KERNEL_SIZE,
                        individual=config.is_individual(model_type)).to(config.DEVICE)
        ckpt = torch.load(cp_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        preds = []
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(config.DEVICE)
                preds.append(model(X).cpu().numpy())
        all_preds.append(np.concatenate(preds, axis=0))
        print(f"  {label}: done")

    # Average predictions across seeds
    preds_ensemble = np.mean(all_preds, axis=0)  # (N_windows, 24, 7)

    # ---- Compute residuals from ensemble predictions ----
    print("\nComputing residuals from ensemble predictions...")
    t_mean = scaler.mean_[n_in:]
    t_scale = scaler.scale_[n_in:]
    residuals = []
    with torch.no_grad():
        for X, Y in test_loader:
            Yt = Y.cpu().numpy()
            residuals.append(Yt)
    all_actuals = np.concatenate(residuals, axis=0)
    # Residuals on scaled scale, then subsample + denormalize
    res_scaled = (all_actuals - preds_ensemble)[::24]  # step=24 for non-overlapping origins
    residuals = res_scaled * t_scale  # denormalize: mean cancels out

    # ---- 2. Ljung-Box (lag 24) ----
    lb_results = ljung_box_test(residuals, target_names, lag=24)

    # ---- 3. Residual normality ----
    residual_normality_check(residuals, target_names, diag_dir)

    # ---- 4. Forecast intervals ----
    print(f"\n{'='*70}")
    print(f" FORECAST INTERVALS (95%) — uncertainty widening over horizon")
    print(f"{'='*70}")
    forecast_intervals(residuals, target_names, diag_dir, ci=95)

    # ---- 5. Bootstrap CI on metrics ----
    bootstrap_ci_metrics(residuals)

    # ---- 6. Stability ----
    is_stable = stability_check(history) if history else True

    # ---- Verdict ----
    diagnostic_verdict(lb_results, all_stationary, is_stable)

    # ---- Generate diagnostic report file ----
    from generate_report import generate_diagnostic_report
    generate_diagnostic_report(model_type)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_diagnostics(sys.argv[1])
    else:
        for mt in ["DLinear-S", "DLinear-I"]:
            run_diagnostics(mt)
