"""
Training script cho DLinear.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_processed_data
from models.dlinear import DLinear

import json, time
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0
    for X, Y in loader:
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, criterion):
    model.eval()
    total = 0
    for X, Y in loader:
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
        total += criterion(model(X), Y).item()
    return total / len(loader)


def plot_training_history(history, save_path):
    """Loss curve: linear + log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = history['val_loss'][best_epoch - 1]

    for ax, log in [(ax1, False), (ax2, True)]:
        if log:
            ax.semilogy(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss', alpha=0.8)
            ax.semilogy(epochs, history['val_loss'], 'r-', lw=2, label='Val Loss', alpha=0.8)
        else:
            ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss', alpha=0.8)
            ax.plot(epochs, history['val_loss'], 'r-', lw=2, label='Val Loss', alpha=0.8)
        ax.scatter(best_epoch, best_val, c='red', s=100, zorder=10,
                   edgecolors='darkred', linewidths=2)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('MSE Loss' + (' (log)' if log else ''), fontweight='bold')
        ax.set_title('DLinear ' + ('Log-' if log else '') + 'Scale', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\nLoss curve -> {save_path}")


def train_model(model_type="DLinear-S", seeds=None):
    """
    Huấn luyện DLinear với model_type cụ thể.

    Args:
        model_type: "DLinear-S" (shared) hoặc "DLinear-I" (individual)
        seeds: None → single seed (config.SEED), list → multi-seed
    """
    if seeds is not None:
        return _multi_seed_train(model_type, seeds)

    individual = config.is_individual(model_type)
    checkpoint_dir = config.get_checkpoint_dir(model_type)

    print("=" * 70)
    print(f" DLinear Weather Forecasting — TRAINING ({model_type})")
    print("=" * 70)
    print(f"Device: {config.DEVICE} | Model: {model_type} | Lookback: {config.LOOKBACK}h | Forecast: {config.FORECAST}h")
    print(f"Kernel: {config.KERNEL_SIZE} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")

    set_seed(config.SEED)

    train_loader, val_loader, test_loader, scaler, feature_names, target_names = load_processed_data()
    n_in = len(feature_names)
    n_out = len(target_names)

    model = DLinear(config.LOOKBACK, config.FORECAST, n_in, out_features=n_out,
                    kernel_size=config.KERNEL_SIZE,
                    individual=individual).to(config.DEVICE)

    print(f"\nParams: {model.count_params():,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    output_dir = config.get_output_dir(model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'Epoch':<8}{'Train Loss':<16}{'Val Loss':<16}{'LR':<12}{'Patience':<10}")
    print("=" * 62)

    t0 = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_epoch(model, val_loader, criterion)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        print(f"{epoch:<8}{train_loss:<16.6f}{val_loss:<16.6f}{lr:<12.2e}{patience_counter}/{config.PATIENCE}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss, 'history': history,
                'feature_names': feature_names, 'target_names': target_names,
                'model_type': model_type,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stop @ epoch {epoch}")
            break

    print(f"\nDone in {time.time()-t0:.1f}s | Best: epoch {best_epoch} val_loss={best_val_loss:.6f}")

    plot_suffix = "_I" if model_type == "DLinear-I" else ""
    plot_training_history(history, os.path.join(output_dir, f'loss_curve{plot_suffix}.png'))
    with open(os.path.join(output_dir, f'training_history{plot_suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history, scaler, feature_names, test_loader


def _multi_seed_train(model_type, seeds=None):
    """
    PROTOCOL §8: Train with multiple seeds, report mean ± std.
    ML/DL are stochastic — single seed is not enough.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    individual = config.is_individual(model_type)
    checkpoint_dir = config.get_checkpoint_dir(model_type)

    results = []
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f" # {model_type} — SEED = {seed}")
        print(f"{'#'*70}")

        config.SEED = seed
        set_seed(seed)

        train_loader, val_loader, test_loader, scaler, feature_names, target_names = load_processed_data()
        n_in = len(feature_names)
        n_out = len(target_names)

        model = DLinear(config.LOOKBACK, config.FORECAST, n_in, out_features=n_out,
                        kernel_size=config.KERNEL_SIZE,
                        individual=individual).to(config.DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, config.EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss = validate_epoch(model, val_loader, criterion)
            lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr)

            if val_loss < best_val_loss:
                best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss, 'history': history,
                    'feature_names': feature_names, 'target_names': target_names,
                    'seed': seed, 'model_type': model_type,
                }, os.path.join(checkpoint_dir, f'best_model_seed{seed}.pth'))
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                break

        results.append({'seed': seed, 'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
                         'history': history})

    # -- Save loss curve + training history (last seed) --
    output_dir = config.get_output_dir(model_type)
    os.makedirs(output_dir, exist_ok=True)
    plot_suffix = "_I" if model_type == "DLinear-I" else ""
    plot_training_history(history, os.path.join(output_dir, f'loss_curve{plot_suffix}.png'))
    with open(os.path.join(output_dir, f'training_history{plot_suffix}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # -- Report mean ± std --
    losses = [r['best_val_loss'] for r in results]
    print(f"\n{'='*60}")
    print(f" {model_type} MULTI-SEED SUMMARY (n={len(results)})")
    print(f"{'='*60}")
    for r in results:
        print(f"  Seed {r['seed']}: val_loss={r['best_val_loss']:.6f} @ epoch {r['best_epoch']}")
    print(f"  Val loss: {np.mean(losses):.6f} ± {np.std(losses):.6f}")

    return results


def train_i():
    """Convenience: train DLinear-I single seed."""
    return train_model(model_type="DLinear-I", seeds=None)


def multi_seed_train_i():
    """Convenience: train DLinear-I with multiple seeds."""
    return train_model(model_type="DLinear-I", seeds=[42, 123, 456])


def train():
    """DLinear-S single seed (backward compat)."""
    return train_model(model_type="DLinear-S", seeds=None)


def multi_seed_train():
    """DLinear-S multi-seed (backward compat)."""
    return train_model(model_type="DLinear-S", seeds=[42, 123, 456])


if __name__ == "__main__":
    # Train both variants with 3 seeds each (Protocol §8)
    multi_seed_train()      # DLinear-S: seeds 42, 123, 456
    multi_seed_train_i()    # DLinear-I: seeds 42, 123, 456
