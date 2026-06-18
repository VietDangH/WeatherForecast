"""
Visualization helpers for DLinear results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prediction_vs_actual(Y_true, Y_pred, feature_idx, feature_name,
                                target_names, window_idx=0, save_path=None):
    """Compare actual vs predicted for one forecast window (24h).
    Y_true, Y_pred are already on real (denormalized) scale — shape (N, H, C)."""

    H = Y_true.shape[1]
    true_vals = Y_true[window_idx, :, feature_idx]
    pred_vals = Y_pred[window_idx, :, feature_idx]

    hours = np.arange(1, H + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    ax1.plot(hours, true_vals, 'b-', lw=2.5, label='Actual', marker='o', ms=4)
    ax1.plot(hours, pred_vals, 'r--', lw=2.5, label='DLinear', marker='s', ms=4)
    ax1.fill_between(hours, true_vals, pred_vals, alpha=0.15, color='gray')
    ax1.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax1.set_ylabel(feature_name, fontweight='bold')
    ax1.set_title(f'DLinear Forecast: {feature_name} (24h ahead)', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    error = true_vals - pred_vals
    colors = ['#d32f2f' if e < 0 else '#1976d2' for e in error]
    ax2.bar(hours, error, color=colors, alpha=0.7, edgecolor='black', lw=0.5)
    ax2.axhline(y=0, color='black', lw=1.5)
    ax2.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
    ax2.set_ylabel(f'Error ({feature_name})', fontweight='bold')
    ax2.set_title('Prediction Error Over Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learned_weights(model, save_path=None):
    """Heatmap of learned Linear weights (trend, seasonal, difference)."""
    if not hasattr(model, 'linear_trend'):
        return

    Wt = model.linear_trend.weight.detach().cpu().numpy()
    Ws = model.linear_seasonal.weight.detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    opts = dict(aspect='auto', cmap='RdBu_r', interpolation='bilinear')
    im1 = ax1.imshow(Wt, **opts); plt.colorbar(im1, ax=ax1)
    ax1.set_title('Trend Branch $W_t$', fontweight='bold')
    ax1.set_xlabel('Past step (120→0h ago)'); ax1.set_ylabel('Forecast step (1→24h)')

    im2 = ax2.imshow(Ws, **opts); plt.colorbar(im2, ax=ax2)
    ax2.set_title('Seasonal Branch $W_s$', fontweight='bold')
    ax2.set_xlabel('Past step'); ax2.set_ylabel('Forecast step')

    im3 = ax3.imshow(Ws - Wt, **opts); plt.colorbar(im3, ax=ax3)
    ax3.set_title('Difference $W_s - W_t$', fontweight='bold')
    ax3.set_xlabel('Past step'); ax3.set_ylabel('Forecast step')

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_bar_chart(results, save_path=None):
    """Horizontal bar chart of MAE per feature."""
    features = list(results.keys())
    mae_vals = [results[f]['MAE'] for f in features]
    idx = np.argsort(mae_vals)[::-1]
    features_sorted = [features[i] for i in idx]
    mae_sorted = [mae_vals[i] for i in idx]

    norm = plt.Normalize(min(mae_sorted), max(mae_sorted))
    colors = plt.cm.RdYlGn_r(norm(mae_sorted))

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(features_sorted, mae_sorted, color=colors, edgecolor='black', lw=0.5)
    for bar, val in zip(bars, mae_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('MAE (lower is better)', fontweight='bold')
    ax.set_title('DLinear Performance by Feature', fontweight='bold')
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
