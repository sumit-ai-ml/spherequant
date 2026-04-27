# CNN Init Rotation Experiment

Small-scale experiment: apply TurboQuant's rotation (SRHT / dense orthogonal)
to CNN kernels and measure whether it improves post-training quantization.

## Why this exists

TurboQuant applies a random orthogonal rotation to unit-norm embeddings so
that each coordinate becomes approximately Beta(d/2, d/2) distributed, which
makes uniform scalar quantization near-optimal. This experiment asks: does
the same trick work on the weights of a trained CNN?

## What's tested

Two hypotheses, both on a 3-layer CNN trained on CIFAR-10.

- **SphereQuant (post-training rotation):** train a standard CNN with Kaiming init,
  then rotate each layer's flattened weight matrix on the fan-in axis before
  quantizing. Compare quantized accuracy vs quantizing unrotated weights.

- **H3 (train in rotated basis):** reparametrize each conv/linear as
  `W = U @ M.T` where `M` is a fixed orthogonal matrix set at init and `U`
  is the trainable parameter. Train U. At quantization time, quantize U
  (not the effective W). Reconstruct W from quantized U.

H1 ("rotate once at init, then train") is a statistical no-op for i.i.d.
Gaussian init (orthogonal rotation of i.i.d. Gaussian = i.i.d. Gaussian)
and is omitted.

## Sweep

- Variants: `baseline`, `spherequant`, `h3_rotated_basis`
- Seeds: 0, 1, 2
- Quantization bits: 2, 4, 6, 8
- Codebooks: uniform grid, Beta(d/2, d/2) Lloyd-Max

## Runtime

~25 min on CPU, ~8 min on GPU. Data cache ~170 MB (CIFAR-10).

## Layout

```
model.py           3-layer CNN with standard Kaiming init
rotated_conv.py    RotatedConv2d and RotatedLinear modules for H3
train.py           10-epoch CIFAR-10 training loop
run.py             orchestrator over seeds x variants x bits
plot.py            figures: acc-vs-bits, weight histograms, Beta KS fit
test_sanity.py     round-trip, norm preservation, Beta-KS, H3 correctness
```

Quantization primitives and rotation utilities live in the top-level
`spherequant` package (`pip install -e .` from the repo root).

## How to run

```bash
# One-time: sanity checks (verifies rotation and H3 plumbing)
python test_sanity.py

# Full sweep (3 seeds x 3 variants + PTQ eval)
python run.py

# Plots from results/results.jsonl
python plot.py
```

Results land in `results/results.jsonl` and `figures/*.png`.
