"""
Systematic comparison of neural network architectures for learning multiplication.
Input: (x1, x2) in [-1M, +1M], Output: x1 * x2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import math
import argparse
from pathlib import Path


# ============================================================
# Dataset
# ============================================================

class MultiplicationDataset(Dataset):
    """Generate random pairs and their product."""
    def __init__(self, n_samples, value_range=1e6, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        self.x1 = rng.uniform(-value_range, value_range, n_samples).astype(np.float64)
        self.x2 = rng.uniform(-value_range, value_range, n_samples).astype(np.float64)
        self.y = self.x1 * self.x2

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        # Normalize: divide by 1e6 so inputs are in [-1, 1], output in [-1, 1]
        x = torch.tensor([self.x1[idx] / 1e6, self.x2[idx] / 1e6], dtype=torch.float32)
        y = torch.tensor([self.y[idx] / 1e12], dtype=torch.float32)  # product of two 1e6 numbers
        return x, y


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
        # Embed each scalar input to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        # Learnable positional embedding for 2 positions
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
        b = x.shape[0]
        # Reshape to (batch, 2, 1) then project
        tokens = self.input_proj(x.unsqueeze(-1))  # (batch, 2, d_model)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        out = self.encoder(tokens)  # (batch, 2, d_model)
        # Pool: mean of both tokens
        pooled = out.mean(dim=1)  # (batch, d_model)
        return self.output_proj(pooled)  # (batch, 1)


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr, device, model_name):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_rel_error': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
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
        val_loss = 0
        val_rel_errors = []
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += nn.functional.mse_loss(pred, y).item()
                # Relative error (avoid div by zero)
                abs_y = torch.abs(y)
                mask = abs_y > 1e-20  # skip near-zero targets
                if mask.any():
                    rel_err = (torch.abs(pred[mask] - y[mask]) / abs_y[mask]).mean().item()
                    val_rel_errors.append(rel_err)
                n_val += 1

        avg_val_loss = val_loss / n_val
        avg_rel_error = np.mean(val_rel_errors) if val_rel_errors else float('inf')

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
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--train-samples', type=int, default=500000)
    parser.add_argument('--val-samples', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--filter', type=str, default=None, help='Only run experiments matching this substring')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Training samples: {args.train_samples}, Val samples: {args.val_samples}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print()

    # Create datasets
    train_ds = MultiplicationDataset(args.train_samples, seed=42)
    val_ds = MultiplicationDataset(args.val_samples, seed=123)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    experiments = get_experiments()
    if args.filter:
        experiments = [(n, m, lr) for n, m, lr in experiments if args.filter in n]

    all_results = {}

    for name, model, lr in experiments:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Training: {name} ({n_params:,} params)")
        print(f"{'='*60}")

        history = train_model(model, train_loader, val_loader, args.epochs, lr, args.device, name)
        all_results[name] = {
            'history': history,
            'n_params': n_params,
            'final_val_mse': history['val_loss'][-1],
            'final_val_rel_error': history['val_rel_error'][-1],
        }

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
    print(f"{'Model':<35} {'Params':>10} {'Val MSE':>12} {'Val RelErr':>12}")
    print("-" * 80)
    for name in sorted(all_results, key=lambda n: all_results[n]['final_val_mse']):
        r = all_results[name]
        print(f"{name:<35} {r['n_params']:>10,} {r['final_val_mse']:>12.2e} {r['final_val_rel_error']:>12.6f}")


if __name__ == '__main__':
    main()
