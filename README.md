# Neural Networks for Multiplication

Systematic comparison of neural network architectures for learning the multiplication function f(x1, x2) = x1 * x2, with inputs sampled from [-1M, +1M].

## Architectures

**MLPs** with various activations:
- ReLU, Tanh, GELU (standard baselines)
- x^2 activation: enables exact multiplication via the identity (a+b)^2 - (a-b)^2 = 4ab
- x|x| activation: smooth, odd function with multiplicative character

**Transformer**: treats each input number as a token. Self-attention's dot-product mechanism (QK^T) inherently performs multiplication, which may help learn this task.

## Usage

```bash
# Train all models
python train.py --epochs 100

# Train specific model
python train.py --epochs 100 --filter square

# Plot results
python plot_results.py
```

## Key Results

The x^2 activation MLP achieves near-perfect precision because it can represent multiplication exactly in 2 layers. Standard activations (ReLU, Tanh) must approximate multiplication and struggle at high precision.
