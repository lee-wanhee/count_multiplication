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

    # ============================================================
    # Figure 4: Input-input ratio vs sequence length
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # The ratio formula:
    # Per layer:
    #   attn input-input = 2 * S^2 * d_model  (QK^T + attn*V)
    #   weight-input     = (2 * d_ff * d_model + 4 * d_model^2) * S  (FFN + QKV/output proj)
    #   ratio = 2*S^2*d / (2*S^2*d + (2*d_ff*d + 4*d^2)*S)
    #         = 2*S / (2*S + 2*d_ff + 4*d)
    #         = S / (S + d_ff + 2*d)
    # Crossover at S = d_ff + 2*d

    seq_lens = np.arange(2, 4097)

    configs = [
        ('ViT-Tiny (d=192, d_ff=768)', 192, 768),
        ('ViT-Small (d=384, d_ff=1536)', 384, 1536),
        ('ViT-Base (d=768, d_ff=3072)', 768, 3072),
        ('ViT-Large (d=1024, d_ff=4096)', 1024, 4096),
    ]

    colors_vit = ['#4daf4a', '#377eb8', '#e41a1c', '#984ea3']

    for (name, d, dff), color in zip(configs, colors_vit):
        ratio = seq_lens / (seq_lens + dff + 2 * d)
        crossover = dff + 2 * d
        axes[0].plot(seq_lens, ratio * 100, label=name, color=color, linewidth=2)
        # Mark crossover
        axes[0].axvline(x=crossover, color=color, linestyle=':', alpha=0.4)
        axes[0].plot(crossover, 50, 'o', color=color, markersize=6)

    # Mark typical ViT sequence lengths for different image resolutions (patch_size=16)
    res_markers = [(64, 16), (128, 64), (256, 256), (512, 1024)]
    for res, n_patches in res_markers:
        axes[0].axvline(x=n_patches+1, color='gray', linestyle='--', alpha=0.3)
        axes[0].text(n_patches+1, 2, f'{res}x{res}\n({n_patches} patches)',
                     fontsize=8, ha='center', color='gray')

    axes[0].axhline(y=50, color='black', linestyle=':', alpha=0.3)
    axes[0].text(50, 52, '50% crossover', fontsize=9, color='gray')

    axes[0].set_xlabel('Sequence Length (S)', fontsize=12)
    axes[0].set_ylabel('Input-Input Multiplications (% of total)', fontsize=12)
    axes[0].set_title('Input-Input Multiplication Ratio vs Sequence Length\n'
                      '(depth-invariant — same ratio at any number of layers)', fontsize=12)
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 75)
    axes[0].set_xlim(2, 5000)

    # Right panel: absolute counts — attention vs FFN per layer
    ax2 = axes[1]

    d_model, d_ff = 768, 3072  # ViT-Base
    seq_range = np.arange(2, 4097)

    attn_per_layer = 2 * seq_range**2 * d_model
    ffn_per_layer = (2 * d_ff * d_model + 4 * d_model**2) * seq_range

    ax2.plot(seq_range, attn_per_layer, color='#e41a1c', linewidth=2,
             label='Attention (input×input)\n$2S^2 d_{model}$')
    ax2.plot(seq_range, ffn_per_layer, color='#377eb8', linewidth=2,
             label='FFN + Projections (weight×input)\n$(2d_{ff}d + 4d^2) \\cdot S$')

    crossover = d_ff + 2 * d_model
    ax2.axvline(x=crossover, color='gray', linestyle=':', alpha=0.5)
    ax2.text(crossover * 1.1, 1e9, f'Crossover\nS={crossover}', fontsize=9, color='gray')

    # Fill regions
    ax2.fill_between(seq_range, attn_per_layer, ffn_per_layer,
                     where=attn_per_layer < ffn_per_layer,
                     alpha=0.1, color='#377eb8', label='_')
    ax2.fill_between(seq_range, attn_per_layer, ffn_per_layer,
                     where=attn_per_layer >= ffn_per_layer,
                     alpha=0.1, color='#e41a1c', label='_')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sequence Length (S)', fontsize=12)
    ax2.set_ylabel('Multiplications per Layer', fontsize=12)
    ax2.set_title('ViT-Base: Attention vs FFN Multiplications\n'
                  '(attention is O(S²), FFN is O(S) — attention dominates at long sequences)',
                  fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/ratio_vs_seqlen.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved ratio_vs_seqlen.png")

    # ============================================================
    # Figure 5: Structural alignment — capacity vs utilization
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # For 256x256 NVS, compare absolute input-input mul counts
    graphics_ii = 109_641_728  # from our analysis

    vit_configs_data = [
        ('ViT-Tiny\n(S=257)', 152_176_896),
        ('ViT-Small\n(S=257)', 608_707_584),
        ('ViT-Base\n(S=257)', 1_217_415_168),
        ('ViT-Large\n(S=257)', 3_246_440_448),
    ]

    labels = ['Graphics\n(targeted)'] + [c[0] for c in vit_configs_data]
    ii_counts = [graphics_ii] + [c[1] for c in vit_configs_data]
    overheads = [c / graphics_ii for c in ii_counts]

    bar_colors = ['#ff7f00'] + ['#377eb8'] * 4
    bars = ax.bar(range(len(labels)), ii_counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Graphics needed line
    ax.axhline(y=graphics_ii, color='#ff7f00', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Graphics needs: {graphics_ii/1e6:.0f}M muls')

    # Annotations
    for i, (bar, oh) in enumerate(zip(bars, overheads)):
        if i == 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    '100% utilized\n(every mul has\ngeometric meaning)',
                    ha='center', fontsize=9, fontweight='bold', color='#ff7f00')
        else:
            waste_pct = (1 - 1/oh) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                    f'{oh:.1f}x capacity\n~{waste_pct:.0f}% structurally\nmisaligned',
                    ha='center', fontsize=9, color='#377eb8')

    ax.set_ylabel('Input-Input Multiplications', fontsize=12)
    ax.set_title('Multiplicative Capacity vs Structural Need\n'
                 'Transformers have ENOUGH multiplications — but most are in the wrong place',
                 fontsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))

    plt.tight_layout()
    plt.savefig('plots/capacity_vs_utilization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved capacity_vs_utilization.png")


if __name__ == '__main__':
    main()
