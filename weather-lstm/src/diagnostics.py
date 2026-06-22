import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from . import config

def run_diagnostics(y_test_real, ensemble_pred):
    print("\n[DIAGNOSTICS: Đang chạy kiểm định Ljung-Box, ADF và Stability trên phần dư h=1]")
    with open(os.path.join(config.OUTPUT_DIR, 'diagnostics.txt'), 'w', encoding='utf-8') as f:
        f.write("LSTM Ensemble Comprehensive Diagnostics (Adaptive LR)\n")
        f.write("="*60 + "\n")
        
        f.write("\n1. LJUNG-BOX TEST\n")
        f.write("-" * 60 + "\n")
        for i, var in enumerate(config.TARGETS):
            res_h1 = (y_test_real[:, 0, i] - ensemble_pred[:, 0, i]).flatten()
            p_val = acorr_ljungbox(res_h1, lags=[24], return_df=True)['lb_pvalue'].values[0]
            status = 'PASS (White Noise)' if p_val > 0.05 else 'FAIL (Autocorrelated)'
            msg = f"{var:>8}: p-value = {p_val:.4f} -> {status}\n"
            print(f"[Ljung-Box] {msg.strip()}")
            f.write(msg)
            
        f.write("\n2. AUGMENTED DICKEY-FULLER TEST\n")
        f.write("-" * 60 + "\n")
        f.write("* H0: Phần dư có nghiệm đơn vị (Không dừng). H1: Phần dư Dừng (Stationary).\n")
        f.write("* Mong muốn: p-value < 0.05 (Dừng) -> Mô hình ổn định theo thời gian.\n\n")
        for i, var in enumerate(config.TARGETS):
            res_h1 = (y_test_real[:, 0, i] - ensemble_pred[:, 0, i]).flatten()
            adf_result = adfuller(res_h1)
            p_val_adf = adf_result[1]
            status_adf = 'PASS (Stationary)' if p_val_adf < 0.05 else 'FAIL (Non-Stationary)'
            msg_adf = f"{var:>8}: p-value = {p_val_adf:.4f} -> {status_adf}\n"
            print(f"[ADF Test]  {msg_adf.strip()}")
            f.write(msg_adf)

        f.write("\n3. RESIDUAL STABILITY CHECK\n")
        f.write("-" * 60 + "\n")
        f.write("* So sánh Mean và Variance của nửa đầu (H1) và nửa sau (H2) tập Test.\n")
        f.write("* Mong muốn: Mean xấp xỉ 0, Variance ít biến động.\n\n")
        for i, var in enumerate(config.TARGETS):
            res_h1 = (y_test_real[:, 0, i] - ensemble_pred[:, 0, i]).flatten()
            mean_all = np.mean(res_h1)
            
            half = len(res_h1) // 2
            mean_1, mean_2 = np.mean(res_h1[:half]), np.mean(res_h1[half:])
            var_1, var_2 = np.var(res_h1[:half]), np.var(res_h1[half:])
            
            msg_stab = (f"{var:>8}: Overall Mean = {mean_all:.4f} | "
                        f"Mean(H1, H2) = ({mean_1:.4f}, {mean_2:.4f}) | "
                        f"Var(H1, H2) = ({var_1:.4f}, {var_2:.4f})\n")
            f.write(msg_stab)

def plot_all_results(y_test_real, ensemble_pred, histories):
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    hz_rmse = [np.sqrt(mean_squared_error(y_test_real[:, h, :].flatten(), ensemble_pred[:, h, :].flatten())) for h in range(config.HORIZON)]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 25), hz_rmse, marker='o', color='#ff7f0e', lw=2.5) 
    plt.title('Per-Horizon RMSE (Ensemble Mean)', fontweight='bold')
    plt.xlabel('Horizon (t+1 to t+24)')
    plt.ylabel('RMSE')
    plt.xticks(range(1, 25))
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'plot_1_horizon_error.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot((y_test_real - ensemble_pred).flatten(), bins=100, kde=True, color='teal')
    plt.title('Residual Normality Histogram', fontweight='bold')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.xlim(-15, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'plot_2_residual_normality.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(16, 20))
    for i, var in enumerate(config.TARGETS):
        plt.subplot(7, 1, i + 1)
        plt.plot(y_test_real[:14, :, i].flatten(), label='Actual', color='#1f77b4', lw=1.5)
        plt.plot(ensemble_pred[:14, :, i].flatten(), label='LSTM Forecast', color='#ff7f0e', lw=1.5, ls='--')
        plt.title(f'Predicted vs Actual: {var.upper()} (First 14 day of 2025)', fontweight='bold')
        plt.ylabel(var)
        if i == 0: plt.legend(loc='upper right')
    plt.xlabel('Time (Hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'plot_3_all_variables.png'), dpi=300)
    plt.close()