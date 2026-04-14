"""
Systematic comparison of neural network architectures for learning multiplication.
Input: (x1, x2) in [-1M, +1M], Output: x1 * x2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
import time


# ============================================================
# Data Generation (all on GPU, no DataLoader overhead)
# ============================================================

def generate_data(n_samples, device, seed=None):
    """Generate normalized multiplication data directly as GPU tensors."""
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(-1, 1, n_samples).astype(np.float32)
    x2 = rng.uniform(-1, 1, n_samples).astype(np.float32)
    y = x1 * x2  # in [-1, 1]

    X = torch.tensor(np.stack([x1, x2], axis=1), device=device)  # (N, 2)
    Y = torch.tensor(y.reshape(-1, 1), device=device)  # (N, 1)
    return X, Y


def batch_iterator(X, Y, batch_size, shuffle=True):
    """Yield batches from pre-loaded GPU tensors."""
    n = X.shape[0]
    if shuffle:
        idx = torch.randperm(n, device=X.device)
        X, Y = X[idx], Y[idx]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]


# ============================================================
# Activation Functions
# ============================================================

class SquareActivation(nn.Module):
    """x^2 activation — enables exact multiplication via (a+b)^2 - (a-b)^2 = 4ab"""
    def forward(self, x):
        return x ** 2


class XAbsX(nn.Module):
    """x * |x| activation — smooth, odd, has multiplicative character"""
    def forward(self, x):
        return x * torch.abs(x)


# ============================================================
# MLP Models
# ============================================================

class MLP(nn.Module):
    def __init__(self, hidden_size, n_layers, activation='relu'):
        super().__init__()
        act_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'gelu': nn.GELU,
            'square': SquareActivation,
            'xabsx': XAbsX,
        }
        act_cls = act_map[activation]

        layers = [nn.Linear(2, hidden_size), act_cls()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), act_cls()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Model
# ============================================================

class TransformerMultiplier(nn.Module):
    """
    Transformer for multiplication. Treats each number as a token.
    Input: 2 tokens, each embedded from scalar to d_model.
    Uses encoder layers, then pools and projects to output.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: (batch, 2)
        tokens = self.input_proj(x.unsqueeze(-1))  # (batch, 2, d_model)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        out = self.encoder(tokens)  # (batch, 2, d_model)
        pooled = out.mean(dim=1)  # (batch, d_model)
        return self.output_proj(pooled)  # (batch, 1)


# ============================================================
# Training
# ============================================================

def train_model(model, train_X, train_Y, val_X, val_Y, epochs, lr, batch_size, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_rel_error': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for x, y in batch_iterator(train_X, train_Y, batch_size, shuffle=True):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            avg_val_loss = nn.functional.mse_loss(val_pred, val_Y).item()
            # Relative error
            abs_y = torch.abs(val_Y)
            mask = abs_y > 1e-10
            if mask.any():
                avg_rel_error = (torch.abs(val_pred[mask] - val_Y[mask]) / abs_y[mask]).mean().item()
            else:
                avg_rel_error = float('inf')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rel_error'].append(avg_rel_error)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} | "
                  f"Train MSE: {avg_train_loss:.2e} | Val MSE: {avg_val_loss:.2e} | "
                  f"Val RelErr: {avg_rel_error:.4f}")

    return history


# ============================================================
# Experiment Configs
# ============================================================

def get_experiments():
    """Return list of (name, model, lr) tuples."""
    experiments = []

    # MLPs with different activations
    for act in ['relu', 'tanh', 'gelu', 'square', 'xabsx']:
        for n_layers in [2, 3]:
            hidden = 128
            name = f"MLP_{act}_L{n_layers}_H{hidden}"
            model = MLP(hidden, n_layers, act)
            experiments.append((name, model, 1e-3))

    # Transformer variants
    for n_layers in [2, 4]:
        for d_model in [64, 128]:
            name = f"Transformer_L{n_layers}_D{d_model}"
            model = TransformerMultiplier(d_model=d_model, nhead=4,
                                          num_layers=n_layers, dim_feedforward=d_model*2)
            experiments.append((name, model, 1e-3))

    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--train-samples', type=int, default=100000)
    parser.add_argument('--val-samples', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--filter', type=str, default=None, help='Only run experiments matching this substring')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Training samples: {args.train_samples}, Val samples: {args.val_samples}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print()

    # Pre-load all data on GPU
    print("Loading data to GPU...")
    train_X, train_Y = generate_data(args.train_samples, args.device, seed=42)
    val_X, val_Y = generate_data(args.val_samples, args.device, seed=123)
    print(f"Data loaded. Train: {train_X.shape}, Val: {val_X.shape}")
    print()

    experiments = get_experiments()
    if args.filter:
        experiments = [(n, m, lr) for n, m, lr in experiments if args.filter in n]

    all_results = {}

    for name, model, lr in experiments:
        model = model.to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Training: {name} ({n_params:,} params)")
        print(f"{'='*60}")

        t0 = time.time()
        history = train_model(model, train_X, train_Y, val_X, val_Y,
                              args.epochs, lr, args.batch_size, name)
        elapsed = time.time() - t0

        all_results[name] = {
            'history': history,
            'n_params': n_params,
            'final_val_mse': history['val_loss'][-1],
            'final_val_rel_error': history['val_rel_error'][-1],
            'training_time_sec': round(elapsed, 1),
        }
        print(f"[{name}] Done in {elapsed:.1f}s | Final Val RelErr: {history['val_rel_error'][-1]:.6f}")

        # Save per-model results
        with open(os.path.join(args.output_dir, f'{name}.json'), 'w') as f:
            json.dump(all_results[name], f, indent=2)

    # Save combined results
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Params':>10} {'Val MSE':>12} {'Val RelErr':>12} {'Time(s)':>8}")
    print("-" * 80)
    for name in sorted(all_results, key=lambda n: all_results[n]['final_val_mse']):
        r = all_results[name]
        print(f"{name:<35} {r['n_params']:>10,} {r['final_val_mse']:>12.2e} {r['final_val_rel_error']:>12.6f} {r['training_time_sec']:>8.1f}")


if __name__ == '__main__':
    main()
