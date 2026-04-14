"""
Analysis: Counting Multiplications in Transformers vs Graphics Pipelines

Motivation: Transformers are increasingly used as "learned graphics engines" for
novel view synthesis (e.g., predicting what a scene looks like from a new camera
pose given a single image). But classical graphics has exact equations that require
specific multiplications. This analysis counts and compares:

1. How many "effective multiplications" a transformer can perform on its inputs
2. How many multiplications the classical graphics pipeline requires
3. Whether transformers are efficient at performing the multiplications needed for graphics

Key insight from our experiments:
- Standard MLP layers (Linear + ReLU/GELU) can only APPROXIMATE multiplication
- Only x^2 activation or attention's QK^T provide TRUE bilinear (multiplicative) operations
- A transformer's multiplicative capacity is finite and countable
"""

import json
import os
import numpy as np


def count_transformer_multiplications(d_model, nhead, num_layers, seq_len, d_ff=None):
    """
    Count the number of distinct input-input multiplications a transformer can perform.

    In a transformer, true multiplications between input-derived quantities occur in:
    1. Attention: QK^T — each head computes a (seq_len x seq_len) matrix of dot products
       Each dot product is a sum of (d_k) element-wise multiplications between
       query and key vectors, which are linear projections of the input.
    2. Attention: attn_weights * V — another bilinear operation mixing attention
       scores with value vectors.

    The FFN layers (Linear + GELU + Linear) do NOT perform input-input multiplications.
    They only compute linear combinations followed by element-wise nonlinearities.
    A linear layer y = Wx + b multiplies WEIGHTS with inputs, not inputs with inputs.
    Only the attention mechanism multiplies input-derived quantities with each other.

    Args:
        d_model: model dimension
        nhead: number of attention heads
        num_layers: number of transformer layers
        seq_len: sequence length (number of tokens)
        d_ff: feedforward dimension (default: 4 * d_model)

    Returns:
        dict with detailed multiplication counts
    """
    if d_ff is None:
        d_ff = 4 * d_model
    d_k = d_model // nhead  # dimension per head

    # Per layer, per head:
    # QK^T: (seq_len x d_k) @ (d_k x seq_len) = seq_len^2 dot products, each of d_k muls
    qk_muls_per_head = seq_len * seq_len * d_k
    # attn @ V: (seq_len x seq_len) @ (seq_len x d_k) = seq_len * d_k dot products, each of seq_len muls
    # But note: after softmax, this is attn_weights * V, which IS a bilinear operation
    # between the attention scores (derived from Q,K which are from input) and V (also from input)
    av_muls_per_head = seq_len * d_k * seq_len

    # Per layer (all heads)
    qk_muls_per_layer = qk_muls_per_head * nhead
    av_muls_per_layer = av_muls_per_head * nhead

    # Total attention multiplications per layer
    attn_muls_per_layer = qk_muls_per_layer + av_muls_per_layer

    # FFN: these are weight-input multiplications, NOT input-input multiplications
    # Linear1: d_model -> d_ff: d_ff * d_model muls (weight * input)
    # Linear2: d_ff -> d_model: d_model * d_ff muls (weight * input)
    # These DON'T count as "effective" multiplications for our analysis
    # because they can only compute linear combinations of inputs
    ffn_weight_muls_per_layer = (d_ff * d_model + d_model * d_ff) * seq_len

    # Total across all layers
    total_attn_muls = attn_muls_per_layer * num_layers
    total_weight_muls = ffn_weight_muls_per_layer * num_layers

    # Also count the projection multiplications (Q, K, V projections)
    # These are weight-input muls, not input-input muls
    qkv_proj_muls = 3 * d_model * d_model * seq_len * num_layers  # Q, K, V projections
    output_proj_muls = d_model * d_model * seq_len * num_layers  # output projection

    return {
        'config': {
            'd_model': d_model, 'nhead': nhead, 'num_layers': num_layers,
            'seq_len': seq_len, 'd_ff': d_ff, 'd_k': d_k,
        },
        'input_input_muls': {
            'qk_per_layer': qk_muls_per_layer,
            'attn_v_per_layer': av_muls_per_layer,
            'total_per_layer': attn_muls_per_layer,
            'total': total_attn_muls,
            'description': 'Multiplications between input-derived quantities (QK^T and attn*V)',
        },
        'weight_input_muls': {
            'ffn_per_layer': ffn_weight_muls_per_layer,
            'qkv_proj_total': qkv_proj_muls,
            'output_proj_total': output_proj_muls,
            'total': total_weight_muls + qkv_proj_muls + output_proj_muls,
            'description': 'Multiplications of learned weights with inputs (linear layers)',
        },
        'total_muls': total_attn_muls + total_weight_muls + qkv_proj_muls + output_proj_muls,
    }


def count_graphics_multiplications(H, W, num_points_per_ray=64):
    """
    Count multiplications in a classical graphics pipeline for novel view synthesis.

    Given: single input image (H x W x 3) + relative camera pose (3x4 matrix)
    Task: render novel view (H x W x 3)

    Classical pipeline (simplified NeRF-like):
    1. Camera ray generation: For each output pixel, compute ray origin + direction
    2. Point sampling: Sample points along each ray
    3. Coordinate transformation: Transform points from one camera frame to another
    4. Color/density lookup: For each 3D point, determine color and density
    5. Volume rendering: Integrate color along each ray

    Args:
        H, W: output image dimensions
        num_points_per_ray: number of sample points per ray

    Returns:
        dict with multiplication counts for each stage
    """
    num_pixels = H * W
    num_rays = num_pixels  # one ray per pixel
    num_points = num_rays * num_points_per_ray

    # Stage 1: Ray generation
    # For each pixel (u, v), compute ray direction d = K^{-1} [u, v, 1]^T
    # K^{-1} is 3x3, so: 3x3 @ 3x1 = 9 muls per pixel
    # Then normalize: 3 muls (squares) + 1 div + 3 muls (scale) = ~7 muls
    ray_gen_muls = num_pixels * (9 + 7)

    # Stage 2: Camera pose transformation
    # For each ray direction: d_world = R @ d_local (3x3 @ 3x1 = 9 muls)
    # For ray origin: just translation (no muls needed beyond the rotation)
    ray_transform_muls = num_rays * 9

    # Stage 3: Point sampling along rays
    # For each sample point: p = o + t * d (3 muls per point for t * d)
    point_sampling_muls = num_points * 3

    # Stage 4: Coordinate transformation (novel view -> reference view)
    # Transform each 3D point using the relative pose: p' = R @ p + t
    # R @ p: 3x3 @ 3x1 = 9 muls per point
    coord_transform_muls = num_points * 9

    # Stage 5: Projection to reference image
    # Project each 3D point to 2D in reference image: p_2d = K @ p' / p'_z
    # K @ p': 3x3 @ 3x1 = 9 muls
    # Perspective division: 2 divisions (= 2 muls with reciprocal)
    projection_muls = num_points * (9 + 2)

    # Stage 6: Bilinear interpolation for color lookup
    # For each projected point, sample the reference image via bilinear interp
    # Bilinear interp: 4 lookups, weights are products of (1-dx)*(1-dy) etc.
    # = 4 weight muls + 4 weighted sums (4 muls each for RGB) = 4 + 12 = 16 muls
    bilinear_muls = num_points * 16

    # Stage 7: Volume rendering (alpha compositing)
    # For each ray, composite N samples:
    # alpha_i = 1 - exp(-sigma_i * delta_i): 1 mul per sample
    # T_i = prod(1 - alpha_j, j<i): 1 mul per sample (running product)
    # C = sum(T_i * alpha_i * c_i): 3 muls (RGB) + 1 mul (T*alpha) = 4 muls per sample
    # Total: 6 muls per sample
    volume_rendering_muls = num_points * 6

    total = (ray_gen_muls + ray_transform_muls + point_sampling_muls +
             coord_transform_muls + projection_muls + bilinear_muls +
             volume_rendering_muls)

    return {
        'config': {'H': H, 'W': W, 'num_points_per_ray': num_points_per_ray},
        'breakdown': {
            'ray_generation': ray_gen_muls,
            'ray_transform': ray_transform_muls,
            'point_sampling': point_sampling_muls,
            'coord_transform': coord_transform_muls,
            'projection': projection_muls,
            'bilinear_interpolation': bilinear_muls,
            'volume_rendering': volume_rendering_muls,
        },
        'total_muls': total,
        'total_input_input_muls': (ray_transform_muls + coord_transform_muls +
                                    projection_muls + volume_rendering_muls),
        'description': (
            'All multiplications involve state variables (camera params, 3D coords, colors). '
            'These are fundamentally input-input multiplications — the camera pose parameters '
            'multiply with ray directions and 3D coordinates. This is exactly the kind of '
            'multiplication that standard neural network layers cannot do efficiently.'
        ),
    }


def count_transformer_for_graphics(H, W, patch_size=16, d_model=768, nhead=12,
                                    num_layers=12, d_ff=3072):
    """
    Count multiplications in a Vision Transformer used for novel view synthesis.

    Architecture: ViT-like encoder that takes image patches + camera pose as input
    and predicts the novel view.

    Args:
        H, W: image dimensions
        patch_size: size of image patches
        d_model: transformer hidden dimension
        nhead: number of attention heads
        num_layers: number of transformer layers
        d_ff: feedforward dimension
    """
    num_patches = (H // patch_size) * (W // patch_size)
    # Add camera pose token(s) — typically 1-4 tokens encoding the 3x4 pose matrix
    num_camera_tokens = 1
    seq_len = num_patches + num_camera_tokens

    # Patch embedding: each patch (patch_size^2 * 3 channels) projected to d_model
    patch_embed_muls = num_patches * (patch_size * patch_size * 3) * d_model

    # Transformer body
    transformer_counts = count_transformer_multiplications(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        seq_len=seq_len, d_ff=d_ff
    )

    # Output head: project from d_model back to patch pixels
    output_muls = num_patches * d_model * (patch_size * patch_size * 3)

    total = patch_embed_muls + transformer_counts['total_muls'] + output_muls

    return {
        'config': {
            'H': H, 'W': W, 'patch_size': patch_size,
            'd_model': d_model, 'nhead': nhead, 'num_layers': num_layers,
            'd_ff': d_ff, 'num_patches': num_patches, 'seq_len': seq_len,
        },
        'patch_embedding_muls': patch_embed_muls,
        'transformer_input_input_muls': transformer_counts['input_input_muls']['total'],
        'transformer_weight_input_muls': transformer_counts['weight_input_muls']['total'],
        'output_head_muls': output_muls,
        'total_muls': total,
        'total_input_input_muls': transformer_counts['input_input_muls']['total'],
    }


def main():
    os.makedirs('results', exist_ok=True)

    print("=" * 80)
    print("ANALYSIS: Multiplication Counting — Transformers vs Graphics Engines")
    print("=" * 80)

    # ============================================================
    # Part 1: Multiplication counts for our experimental models
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 1: Our Experimental Models (seq_len=2)")
    print("=" * 60)

    configs = [
        ('Transformer_L2_D64', 64, 4, 2, 2),
        ('Transformer_L2_D128', 128, 4, 2, 2),
        ('Transformer_L4_D64', 64, 4, 4, 2),
        ('Transformer_L4_D128', 128, 4, 4, 2),
    ]

    for name, d_model, nhead, num_layers, seq_len in configs:
        counts = count_transformer_multiplications(d_model, nhead, num_layers, seq_len)
        print(f"\n{name}:")
        print(f"  Input-input muls (attention): {counts['input_input_muls']['total']:>12,}")
        print(f"  Weight-input muls (linear):   {counts['weight_input_muls']['total']:>12,}")
        print(f"  Total:                         {counts['total_muls']:>12,}")
        print(f"  Ratio (input-input / total):   {counts['input_input_muls']['total']/counts['total_muls']:.4f}")

    # ============================================================
    # Part 2: Graphics pipeline multiplication count
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2: Classical Graphics Pipeline")
    print("=" * 60)

    for res in [(64, 64), (128, 128), (256, 256)]:
        H, W = res
        for n_pts in [64, 128]:
            g = count_graphics_multiplications(H, W, n_pts)
            print(f"\n  Resolution {H}x{W}, {n_pts} pts/ray:")
            for stage, count in g['breakdown'].items():
                print(f"    {stage:<30s}: {count:>12,}")
            print(f"    {'TOTAL':<30s}: {g['total_muls']:>12,}")
            print(f"    Input-input muls:             {g['total_input_input_muls']:>12,}")

    # ============================================================
    # Part 3: Transformer for graphics — ViT-style
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: Transformer as Graphics Engine")
    print("=" * 60)

    vit_configs = [
        ('ViT-Tiny (NVS)',   256, 256, 16, 192,  3,  6,  768),
        ('ViT-Small (NVS)',  256, 256, 16, 384,  6, 12, 1536),
        ('ViT-Base (NVS)',   256, 256, 16, 768, 12, 12, 3072),
        ('ViT-Large (NVS)',  256, 256, 16, 1024, 16, 24, 4096),
    ]

    for name, H, W, ps, d, nh, nl, dff in vit_configs:
        t = count_transformer_for_graphics(H, W, ps, d, nh, nl, dff)
        print(f"\n  {name} ({t['config']['num_patches']} patches, seq_len={t['config']['seq_len']}):")
        print(f"    Patch embedding:              {t['patch_embedding_muls']:>15,}")
        print(f"    Attn input-input muls:        {t['total_input_input_muls']:>15,}")
        print(f"    Weight-input muls:            {t['transformer_weight_input_muls']:>15,}")
        print(f"    Output head:                  {t['output_head_muls']:>15,}")
        print(f"    TOTAL:                        {t['total_muls']:>15,}")

    # ============================================================
    # Part 4: Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 4: Efficiency Comparison")
    print("=" * 60)

    H, W = 256, 256
    n_pts = 64

    graphics = count_graphics_multiplications(H, W, n_pts)

    print(f"\nTask: Novel view synthesis at {H}x{W}")
    print(f"Classical graphics ({n_pts} pts/ray):")
    print(f"  Total multiplications:     {graphics['total_muls']:>15,}")
    print(f"  Input-input muls:          {graphics['total_input_input_muls']:>15,}")
    print(f"  (All muls involve scene/camera state variables)")

    for name, H, W, ps, d, nh, nl, dff in vit_configs:
        t = count_transformer_for_graphics(H, W, ps, d, nh, nl, dff)
        ratio_total = t['total_muls'] / graphics['total_muls']
        ratio_ii = t['total_input_input_muls'] / graphics['total_input_input_muls']

        print(f"\n  {name}:")
        print(f"    Total muls:              {t['total_muls']:>15,}")
        print(f"    Input-input muls:        {t['total_input_input_muls']:>15,}")
        print(f"    Total overhead vs graphics:    {ratio_total:>10.1f}x")
        print(f"    Input-input overhead:          {ratio_ii:>10.1f}x")
        print(f"    Fraction of muls that are input-input: "
              f"{t['total_input_input_muls']/t['total_muls']:.4f}")

    # ============================================================
    # Part 5: Key Insight
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 5: Key Insight")
    print("=" * 60)
    print("""
Classical graphics performs TARGETED multiplications:
  - Each multiplication has a specific geometric meaning
  - Ray direction * camera rotation = world-space ray (9 muls)
  - 3D point * projection matrix = 2D pixel coordinate (9 muls)
  - Alpha * transmittance * color = rendered pixel (4 muls)

Transformers perform GENERIC multiplications via attention:
  - QK^T multiplies ALL pairs of token representations
  - Most of these multiplications are "wasted" — they compute relationships
    between pairs of patches that have no geometric relevance
  - Only a tiny fraction of the attention multiplications correspond to
    the actual geometric operations needed

The efficiency gap:
  - Graphics: O(H*W*N) multiplications, each geometrically meaningful
  - Transformer: O(S^2 * d * L) attention muls, where S = (H/P)^2
    Most are not geometrically meaningful — the network must LEARN
    which multiplications to use from the exponentially large space

This explains why:
  1. Transformers need massive scale to match graphics quality
  2. But they CAN learn graphics — attention provides the multiplicative
     operations needed, just very inefficiently
  3. The x^2 activation result shows that architectures with the RIGHT
     inductive bias (matching the structure of the target function)
     can be exponentially more efficient
""")

    # Save results
    results = {
        'graphics_256x256_64pts': count_graphics_multiplications(256, 256, 64),
        'transformer_configs': {},
    }
    for name, H, W, ps, d, nh, nl, dff in vit_configs:
        results['transformer_configs'][name] = count_transformer_for_graphics(H, W, ps, d, nh, nl, dff)

    with open('results/multiplication_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/multiplication_analysis.json")


if __name__ == '__main__':
    main()
