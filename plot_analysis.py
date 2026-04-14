"""Plot the multiplication counting analysis: Transformers vs Graphics Engines."""

import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    os.makedirs('plots', exist_ok=True)

    # ============================================================
    # Figure 1: Comparison bar chart
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Data for 256x256
    models = ['Graphics\n(64 pts/ray)', 'ViT-Tiny', 'ViT-Small', 'ViT-Base', 'ViT-Large']
    total_muls = np.array([190_382_080, 909_805_824, 6_216_754_176, 23_347_611_648, 81_260_494_848])
    ii_muls = np.array([109_641_728, 152_176_896, 608_707_584, 1_217_415_168, 3_246_440_448])
    wi_muls = total_muls - ii_muls

    x = np.arange(len(models))
    width = 0.6

    # Left: total multiplications (log scale)
    bars1 = axes[0].bar(x, ii_muls, width, label='Input-Input muls\n(attention QK^T, attn*V)', color='#e41a1c')
    bars2 = axes[0].bar(x, wi_muls, width, bottom=ii_muls, label='Weight-Input muls\n(linear layers)', color='#377eb8', alpha=0.7)

    axes[0].set_yscale('log')
    axes[0].set_ylabel('Number of Multiplications')
    axes[0].set_title('Total Multiplications for 256x256 Novel View Synthesis')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add overhead labels
    for i in range(1, len(models)):
        overhead = total_muls[i] / total_muls[0]
        axes[0].text(i, total_muls[i] * 1.3, f'{overhead:.0f}x', ha='center', fontsize=10, fontweight='bold')

    # Right: fraction of input-input muls
    fractions = ii_muls / total_muls
    colors = ['#ff7f00'] + ['#4daf4a'] * 4
    bars = axes[1].bar(x, fractions * 100, width, color=colors)

    axes[1].set_ylabel('Input-Input Multiplications (%)')
    axes[1].set_title('Fraction of "True" Multiplications\n(input × input, not weight × input)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, frac in zip(bars, fractions):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{frac*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/transformer_vs_graphics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved transformer_vs_graphics.png")

    # ============================================================
    # Figure 2: Scaling analysis
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    resolutions = [64, 128, 256, 512]
    pts_per_ray = 64

    # Graphics pipeline scaling
    graphics_muls = []
    for r in resolutions:
        total = r * r * (16 + 9) + r * r * pts_per_ray * (3 + 9 + 11 + 16 + 6)
        graphics_muls.append(total)

    # Transformer scaling (ViT-Base, patch_size=16)
    d_model, nhead, num_layers, d_ff = 768, 12, 12, 3072
    transformer_muls = []
    transformer_ii_muls = []
    for r in resolutions:
        n_patches = (r // 16) ** 2
        seq_len = n_patches + 1
        # Patch embedding + output
        embed = 2 * n_patches * (16 * 16 * 3) * d_model
        # Attention input-input
        d_k = d_model // nhead
        attn_ii = 2 * seq_len * seq_len * d_k * nhead * num_layers
        # Weight-input (FFN + projections)
        weight = (2 * d_ff * d_model + 4 * d_model * d_model) * seq_len * num_layers
        transformer_muls.append(embed + attn_ii + weight)
        transformer_ii_muls.append(attn_ii)

    ax.plot(resolutions, graphics_muls, 'o-', color='#e41a1c', linewidth=2, markersize=8,
            label=f'Graphics pipeline ({pts_per_ray} pts/ray)')
    ax.plot(resolutions, transformer_muls, 's-', color='#377eb8', linewidth=2, markersize=8,
            label='ViT-Base (total muls)')
    ax.plot(resolutions, transformer_ii_muls, 's--', color='#377eb8', linewidth=2, markersize=8,
            alpha=0.5, label='ViT-Base (input-input only)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image Resolution (H = W)', fontsize=12)
    ax.set_ylabel('Number of Multiplications', fontsize=12)
    ax.set_title('Scaling: Multiplications vs Resolution', fontsize=14)
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate scaling
    ax.annotate('Graphics: O(H²·N)\nlinear in pixels',
                xy=(256, graphics_muls[2]), xytext=(128, graphics_muls[2] * 20),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax.annotate('Transformer: O(S²·d·L)\nS = (H/P)² patches\nquadratic in patches',
                xy=(256, transformer_muls[2]), xytext=(400, transformer_muls[2] * 0.02),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('plots/scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved scaling_analysis.png")

    # ============================================================
    # Figure 3: The multiplication efficiency story
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # From our experiments
    models_exp = ['MLP\n(ReLU)', 'MLP\n(Tanh)', 'MLP\n(GELU)', 'MLP\n(x|x|)',
                  'Transformer\n(L2 D64)', 'MLP\n(x²)']
    rel_errors = [0.048878, 0.089716, 0.022223, 0.022887, 0.012820, 0.000003]
    ii_mul_ratios = [0, 0, 0, 0, 0.0052, 1.0]  # fraction of ops that are true input-input muls

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ff7f00']

    ax2 = ax.twinx()

    x = np.arange(len(models_exp))
    bars = ax.bar(x - 0.2, rel_errors, 0.35, color=colors, alpha=0.8, label='Relative Error')
    dots = ax2.bar(x + 0.2, [r * 100 for r in ii_mul_ratios], 0.35,
                   color='gray', alpha=0.4, label='Input-Input Mul %')

    ax.set_yscale('log')
    ax.set_ylabel('Validation Relative Error (log scale)', fontsize=11)
    ax2.set_ylabel('Input-Input Multiplications (% of total ops)', fontsize=11)
    ax.set_title('Correlation: Multiplicative Capacity vs Precision on f(x,y) = x·y', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models_exp, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('plots/multiplication_vs_precision.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved multiplication_vs_precision.png")


if __name__ == '__main__':
    main()
