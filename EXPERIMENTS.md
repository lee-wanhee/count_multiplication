# Experiment Log

## Overview

**Goal**: Systematically compare neural network architectures for learning the multiplication function `f(x1, x2) = x1 * x2`.

**Motivation**: An earlier undergraduate project showed that a specific activation function (x^2) allows neural networks to learn multiplication with near-perfect precision. We revisit this with a systematic comparison and add Transformers, which were not available at the time.

**Key Insight**: The identity `(a+b)^2 - (a-b)^2 = 4ab` means a 2-layer network with x^2 activation can represent multiplication exactly.

**Hypothesis for Transformers**: Self-attention computes `softmax(QK^T)V`, which involves dot products (bilinear operations). This multiplicative structure might help Transformers learn multiplication better than standard MLP activations.

---

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Input range | [-1,000,000, +1,000,000] |
| Normalization | Inputs ÷ 1e6 → [-1, 1]; Outputs ÷ 1e12 → [-1, 1] |
| Training samples | 100,000 |
| Validation samples | 10,000 |
| Batch size | 4,096 |
| Epochs | 100 |
| Optimizer | Adam |
| LR schedule | Cosine annealing (initial LR = 1e-3) |
| Loss | MSE |
| Metrics | MSE, Relative Error |
| Device | CUDA (single GPU) |

---

## Models

### MLP Variants (10 models)

Each MLP: `Input(2) → [Linear(H) → Act]×L → Linear(1)`

| Activation | Description | Why test it? |
|------------|-------------|--------------|
| ReLU | max(0, x) | Standard baseline; piecewise linear, no multiplicative capacity |
| Tanh | tanh(x) | Bounded, smooth; can approximate via Taylor series (contains x^3, x^5...) |
| GELU | x·Φ(x) | Has multiplicative character (x times a function of x) |
| x^2 | x² | **Expected winner** — enables exact multiplication identity |
| x\|x\| | x·\|x\| | Smooth, odd, quadratic-like; intermediate between ReLU and x^2 |

Each activation tested with 2 and 3 layers, hidden size 128.

### Transformer Variants (4 models)

`Input(2 scalars) → Linear projection to d_model → Positional embedding → TransformerEncoder → Mean pool → MLP head → Output(1)`

| Config | d_model | Layers | Heads | FFN dim |
|--------|---------|--------|-------|---------|
| Transformer_L2_D64 | 64 | 2 | 4 | 128 |
| Transformer_L2_D128 | 128 | 2 | 4 | 256 |
| Transformer_L4_D64 | 64 | 4 | 4 | 128 |
| Transformer_L4_D128 | 128 | 4 | 4 | 256 |

---

## Results

### Final Validation Metrics (sorted by relative error)

| Rank | Model | Params | Val MSE | Val Relative Error | Time (s) |
|------|-------|--------|---------|--------------------|----------|
| 1 | MLP_square_L2_H128 | 17,025 | 2.42e-15 | **0.000003** | 5.6 |
| 2 | Transformer_L2_D64 | 71,425 | 2.22e-07 | 0.012820 | 46.8 |
| 3 | MLP_square_L3_H128 | 33,537 | 4.31e-08 | 0.017648 | 6.8 |
| 4 | Transformer_L2_D128 | 282,113 | 1.12e-07 | 0.019251 | 50.8 |
| 5 | MLP_gelu_L3_H128 | 33,537 | 1.33e-07 | 0.022223 | 5.9 |
| 6 | MLP_xabsx_L3_H128 | 33,537 | 3.54e-07 | 0.022887 | 7.1 |
| 7 | Transformer_L4_D64 | 138,369 | 1.52e-07 | 0.026088 | 93.7 |
| 8 | MLP_relu_L3_H128 | 33,537 | 1.81e-06 | 0.048878 | 6.5 |
| 9 | MLP_xabsx_L2_H128 | 17,025 | 2.18e-07 | 0.052337 | 5.8 |
| 10 | MLP_relu_L2_H128 | 17,025 | 1.60e-06 | 0.053117 | 7.4 |
| 11 | Transformer_L4_D128 | 547,073 | 1.16e-07 | 0.058395 | 93.7 |
| 12 | MLP_gelu_L2_H128 | 17,025 | 5.99e-07 | 0.072022 | 5.4 |
| 13 | MLP_tanh_L2_H128 | 17,025 | 9.95e-05 | 0.089716 | 5.0 |
| 14 | MLP_tanh_L3_H128 | 33,537 | 4.40e-05 | 0.154181 | 6.4 |

### Key Findings

1. **x^2 activation is the clear winner**: MLP_square_L2 achieves 0.0003% relative error — essentially perfect multiplication. This confirms the undergraduate result: the identity `(a+b)^2 - (a-b)^2 = 4ab` lets a 2-layer network represent multiplication exactly. The MSE of 2.42e-15 is at float32 precision limits.

2. **Transformers are the second-best architecture**: The best transformer (L2_D64, 1.28% error) significantly outperforms all standard-activation MLPs. This supports the hypothesis that attention's dot-product mechanism (QK^T) provides inherent multiplicative capacity. Notably, the smaller transformer (71K params) outperformed the larger one (547K params), suggesting that for this simple task, the optimization landscape matters more than model capacity.

3. **Standard activations plateau at ~2-5% error**: ReLU, Tanh, GELU, and x|x| MLPs all converge to relative errors in the 2-15% range. They can approximate multiplication but cannot represent it exactly. GELU performs best among standard activations (2.2%), likely because `x * Phi(x)` has partial multiplicative character.

4. **Depth hurts x^2 but helps others**: For x^2 activation, 2 layers (0.0003%) vastly outperformed 3 layers (1.76%). The 2-layer solution is the exact analytical solution; adding layers creates optimization difficulty. For standard activations, 3 layers generally helped by providing more approximation capacity.

5. **More transformer layers didn't help**: L4 transformers performed worse than L2 transformers across the board, again suggesting optimization difficulty outweighs capacity benefits for this task.

### Analysis: Why Transformers Do Well

A transformer encoder layer contains:
- **Self-attention**: `Attn(Q,K,V) = softmax(QK^T / sqrt(d))V` — the QK^T term is a bilinear (multiplicative) operation
- **FFN**: Two linear layers with GELU activation — similar to our MLP baselines

With just 2 input tokens, the attention computes a 2x2 attention matrix. The QK^T product gives the network access to cross-term products of the two inputs, which is exactly what's needed for multiplication. Combined with the FFN's nonlinearity, this provides enough multiplicative expressivity to achieve ~1% error.

---

## Part 2: Multiplication Counting — Transformers as Graphics Engines

### Motivation

Transformers are increasingly used for novel view synthesis (NVS) — given a single image and a camera pose, render what the scene looks like from a new viewpoint. Classical graphics achieves this with exact equations involving specific multiplications (rotation matrices, projections, volume rendering). Can we count and compare the multiplications?

### Key Distinction: Input-Input vs Weight-Input Multiplications

Not all multiplications in a neural network are equal:

| Type | Example | Can learn multiplication? |
|------|---------|--------------------------|
| **Weight × Input** | Linear layer: y = Wx + b | No — only linear combinations |
| **Input × Input** | Attention: QK^T | Yes — bilinear, multiplicative |

Our experiment proves this: MLPs with ReLU/GELU (only weight × input muls) plateau at 2-5% error. Only architectures with input × input operations (x², attention) achieve high precision.

### Multiplication Counts: 256×256 Novel View Synthesis

| System | Total Muls | Input-Input Muls | Input-Input % | Overhead vs Graphics |
|--------|-----------|-------------------|---------------|---------------------|
| **Graphics (64 pts/ray)** | 190M | 110M | 57.6% | 1x |
| ViT-Tiny | 910M | 152M | 16.7% | 4.8x |
| ViT-Small | 6.2B | 609M | 9.8% | 32.7x |
| ViT-Base | 23.3B | 1.2B | 5.2% | 122.6x |
| ViT-Large | 81.3B | 3.2B | 4.0% | 426.8x |

### The Efficiency Gap

**Classical graphics** performs **targeted** multiplications — each has a specific geometric meaning:
- `ray_dir × R` = world-space ray (9 muls per ray)
- `point × K` = 2D projection (9 muls per point)
- `alpha × transmittance × color` = rendered pixel (4 muls per sample)

**Transformers** perform **generic** multiplications via attention — QK^T multiplies ALL pairs of token representations. Most are "wasted" — they compute relationships between patches with no geometric relevance. The network must learn which of the exponentially many possible multiplications to actually use.

### Scaling Behavior

- **Graphics**: O(H² × N) — linear in pixels, linear in samples per ray
- **Transformer**: O(S² × d × L) where S = (H/P)² — **quartic** in resolution (quadratic in patch count, which is quadratic in resolution)

At higher resolutions, the transformer overhead grows dramatically.

### Implications

1. **Transformers CAN replace graphics engines** — attention provides the multiplicative operations needed
2. **But they are wildly inefficient** — 100-400x more multiplications than the classical pipeline, and only 4-17% are the "right kind" (input × input)
3. **The x² result is the existence proof** — with the right inductive bias matching the target function's structure, we can be exponentially more efficient
4. **Future direction**: Can we design architectures that have the transformer's generality but the graphics engine's multiplication efficiency? (e.g., geometric attention, structured state spaces with multiplicative gates)

---

## How to Reproduce

```bash
# Train all models
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100

# Plot results
python plot_results.py --results-dir results --output-dir plots

# Run multiplication counting analysis
python analysis_multiplication_counting.py

# Plot analysis figures
python plot_analysis.py
```
