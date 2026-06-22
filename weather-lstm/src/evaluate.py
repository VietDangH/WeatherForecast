import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from . import config

def safe_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=1000):
    np.random.seed(42)
    n_windows = y_true.shape[0]
    mae_list, rmse_list, mape_list, r2_list = [], [], [], []
    for _ in range(n_bootstraps):
        idx = np.random.randint(0, n_windows, size=n_windows)
        yt, yp = y_true[idx].flatten(), y_pred[idx].flatten()
        mae_list.append(mean_absolute_error(yt, yp))
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
        mape_list.append(safe_mape(yt, yp))
        r2_list.append(r2_score(yt, yp))
    return [np.percentile(lst, [2.5, 97.5]) for lst in (mae_list, rmse_list, mape_list, r2_list)]

def evaluate_metrics(y_test_real, ensemble_pred):
    yt_f, yp_f = y_test_real.flatten(), ensemble_pred.flatten()
    mae_ci, rmse_ci, mape_ci, r2_ci = compute_bootstrap_ci(y_test_real, ensemble_pred)

    print(f"\n=> [OVERALL ENSEMBLE] MAE : {mean_absolute_error(yt_f, yp_f):.3f} [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
    print(f"=> [OVERALL ENSEMBLE] RMSE: {np.sqrt(mean_squared_error(yt_f, yp_f)):.3f} [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]")
    print(f"=> [OVERALL ENSEMBLE] MAPE: {safe_mape(yt_f, yp_f):.3f}% [{mape_ci[0]:.3f}%, {mape_ci[1]:.3f}%]")
    print(f"=> [OVERALL ENSEMBLE] R2  : {r2_score(yt_f, yp_f):.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

    var_metrics = []
    for i, var in enumerate(config.TARGETS):
        yt_v, yp_v = y_test_real[:, :, i].flatten(), ensemble_pred[:, :, i].flatten()
        m_ci, r_ci, mp_ci, r2_ci_v = compute_bootstrap_ci(y_test_real[:, :, i], ensemble_pred[:, :, i])
        var_metrics.append({
            'Variable': var,
            'MAE': f"{mean_absolute_error(yt_v, yp_v):.3f} [{m_ci[0]:.3f}, {m_ci[1]:.3f}]",
            'RMSE': f"{np.sqrt(mean_squared_error(yt_v, yp_v)):.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]",
            'MAPE (%)': f"{safe_mape(yt_v, yp_v):.3f}% [{mp_ci[0]:.3f}%, {mp_ci[1]:.3f}%]",
            'R2 Score': f"{r2_score(yt_v, yp_v):.4f} [{r2_ci_v[0]:.4f}, {r2_ci_v[1]:.4f}]"
        })

    df_var = pd.DataFrame(var_metrics)
    df_var.to_csv(os.path.join(config.OUTPUT_DIR, 'per_variable_metrics.csv'), index=False)
    print("\n[BẢNG SAI SỐ CHI TIẾT TỪNG BIẾN - Kèm 95% CI]")
    print(df_var.to_string(index=False))