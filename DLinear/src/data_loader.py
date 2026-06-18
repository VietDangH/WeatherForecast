"""
Data Loader — load processed data từ outputs/.
Theo protocol: input (B, 120, 12) -> target (B, 24, 7).
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config


class WeatherDataset(Dataset):
    """Sliding-window dataset with separate X (input) and Y (target) arrays."""

    def __init__(self, X, Y, lookback, forecast):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.lookback = lookback
        self.forecast = forecast

    def __len__(self):
        # X[t:t+L] -> Y[t+L:t+L+H]
        return len(self.X) - self.lookback - self.forecast + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.lookback]                              # (L, C_in)
        y = self.Y[idx + self.lookback : idx + self.lookback + self.forecast]  # (H, C_out)
        return x, y


def load_processed_data():
    """Load processed .npz, scaler, feature/target names -> PyTorch DataLoaders."""

    print("=" * 70)
    print(" LOADING PROCESSED DATA (Protocol: 120h -> 24h)")
    print("=" * 70)

    print("[1/4] Loading processed data...")
    data = np.load(config.PROCESSED_DATA_PATH)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    Y_train, Y_val, Y_test = data['Y_train'], data['Y_val'], data['Y_test']

    print("[2/4] Loading scaler...")
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("[3/4] Loading feature & target names...")
    with open(config.FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    with open(config.TARGET_NAMES_PATH) as f:
        target_names = json.load(f)

    n_in = len(feature_names)
    n_out = len(target_names)
    print(f"    Input: {n_in} features | Target: {n_out} variables")
    print(f"    Train: {X_train.shape[0]:,}h | Val: {X_val.shape[0]:,}h | Test: {X_test.shape[0]:,}h")

    print("[4/4] Creating sliding windows...")
    train_ds = WeatherDataset(X_train, Y_train, config.LOOKBACK, config.FORECAST)
    val_ds   = WeatherDataset(X_val,   Y_val,   config.LOOKBACK, config.FORECAST)
    test_ds  = WeatherDataset(X_test,  Y_test,  config.LOOKBACK, config.FORECAST)

    print(f"    Train windows: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    # Shape check
    X, Y = next(iter(train_loader))
    print(f"\n{'-'*50}\nShape check:")
    print(f"  X: {tuple(X.shape)}  <- expected ({config.BATCH_SIZE}, {config.LOOKBACK}, {n_in})")
    print(f"  Y: {tuple(Y.shape)}  <- expected ({config.BATCH_SIZE}, {config.FORECAST}, {n_out})")
    assert X.shape == (config.BATCH_SIZE, config.LOOKBACK, n_in), f"X: {X.shape}"
    assert Y.shape == (config.BATCH_SIZE, config.FORECAST, n_out), f"Y: {Y.shape}"
    print("  OK!")

    return train_loader, val_loader, test_loader, scaler, feature_names, target_names


if __name__ == "__main__":
    load_processed_data()
