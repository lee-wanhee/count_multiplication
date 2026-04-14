"""Plot results from the multiplication learning experiments."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load all individual result files."""
    results = {}
    for f in sorted(Path(results_dir).glob('*.json')):
        if f.name == 'all_results.json':
            continue
        name = f.stem
        with open(f) as fh:
            results[name] = json.load(fh)
    return results


def categorize(name):
    """Return (architecture, activation/config) for grouping."""
    if name.startswith('Transformer'):
        return 'Transformer', name
    parts = name.split('_')
    return 'MLP', parts[1]  # activation name


def plot_all(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    act_colors = {
        'relu': '#e41a1c',
        'tanh': '#377eb8',
        'gelu': '#4daf4a',
        'square': '#ff7f00',
        'xabsx': '#984ea3',
    }
    transformer_colors = ['#a65628', '#f781bf', '#999999', '#66c2a5']

    # ============================================================
    # 1. Training loss curves (all models)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, r in sorted(results.items()):
        arch, act = categorize(name)
        if arch == 'MLP':
            color = act_colors.get(act, 'black')
            linestyle = '--' if 'L3' in name else '-'
        else:
            idx = sorted([n for n in results if n.startswith('Transformer')]).index(name)
            color = transformer_colors[idx % len(transformer_colors)]
            linestyle = '-'

        epochs = range(1, len(r['history']['train_loss']) + 1)
        axes[0].plot(epochs, r['history']['train_loss'], label=name, color=color, linestyle=linestyle, alpha=0.8)
        axes[1].plot(epochs, r['history']['val_rel_error'], label=name, color=color, linestyle=linestyle, alpha=0.8)

    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train MSE (log)')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Relative Error (log)')
    axes[1].set_title('Validation Relative Error')
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss_curves.png")

    # ============================================================
    # 2. Bar chart: final val relative error by model
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_names = sorted(results.keys(), key=lambda n: results[n]['final_val_rel_error'])
    colors = []
    for name in sorted_names:
        arch, act = categorize(name)
        if arch == 'MLP':
            colors.append(act_colors.get(act, 'black'))
        else:
            idx = sorted([n for n in results if n.startswith('Transformer')]).index(name)
            colors.append(transformer_colors[idx % len(transformer_colors)])

    vals = [results[n]['final_val_rel_error'] for n in sorted_names]
    bars = ax.bar(range(len(sorted_names)), vals, color=colors)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Final Validation Relative Error')
    ax.set_title('Final Precision Comparison (lower is better)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2e}', ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved final_comparison.png")

    # ============================================================
    # 3. Activation type comparison (grouped bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by activation type
    act_groups = {}
    transformer_results = {}
    for name, r in results.items():
        arch, act = categorize(name)
        if arch == 'MLP':
            if act not in act_groups:
                act_groups[act] = []
            act_groups[act].append((name, r['final_val_rel_error']))
        else:
            transformer_results[name] = r['final_val_rel_error']

    # Best per activation
    labels = []
    best_vals = []
    bar_colors = []
    for act in ['relu', 'tanh', 'gelu', 'square', 'xabsx']:
        if act in act_groups:
            best = min(act_groups[act], key=lambda x: x[1])
            labels.append(f"MLP ({act})\n{best[0]}")
            best_vals.append(best[1])
            bar_colors.append(act_colors[act])

    for name, val in sorted(transformer_results.items(), key=lambda x: x[1]):
        labels.append(f"Transformer\n{name}")
        best_vals.append(val)
        idx = sorted(transformer_results.keys()).tolist().index(name) if hasattr(sorted(transformer_results.keys()), 'tolist') else list(sorted(transformer_results.keys())).index(name)
        bar_colors.append(transformer_colors[idx % len(transformer_colors)])

    bars = ax.bar(range(len(labels)), best_vals, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Best Validation Relative Error')
    ax.set_title('Best Model per Architecture/Activation Type')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved activation_comparison.png")

    # ============================================================
    # 4. Summary table
    # ============================================================
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (sorted by relative error)")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Params':>10} {'Val MSE':>12} {'Val RelErr':>12}")
    print("-" * 80)
    for name in sorted(results, key=lambda n: results[n]['final_val_rel_error']):
        r = results[name]
        print(f"{name:<35} {r['n_params']:>10,} {r['final_val_mse']:>12.2e} {r['final_val_rel_error']:>12.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        exit(1)

    plot_all(results, args.output_dir)
