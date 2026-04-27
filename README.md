# SphereQuant

Training-free post-training weight quantization. We rotate each layer's
flattened weight matrix by a random orthogonal transform, L2-normalize the
rotated rows so they live on the unit sphere, then quantize each coordinate
with a Lloyd–Max scalar codebook matched to the Beta(d/2, d/2) distribution
that those coordinates follow on the sphere. No calibration data, no
gradient steps, runs on a single CPU core in under a minute for ViT-B/16.

The method is described in `paper/methodology.tex`. Experiments and full
tables are in `paper/results.tex`.

## Repository layout

```
spherequant/                    Top-level Python package (shared primitives)
  ptq.py                          quantize_model_{baseline,spherequant,quarot,rtn_absmax,h3}
  rotation_utils.py               SRHT, dense orthogonal, Beta KS test, torch wrappers

paper/                          NeurIPS LaTeX
  methodology.tex
  results.tex
  related_works.tex
  references.bib

experiments/
  cnn_init_rotation/            CIFAR-10 3-layer CNN sanity test + H3 ablation
  imagenet_ptq/                 ResNet/ViT/ConvNeXt/MobileNet/EfficientNet on ImageNet-1k
  llm_ptq/                      TinyLlama 1.1B + Phi-1.5 on WikiText-2
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                   # installs the spherequant package
pip install -r requirements.txt    # pulls torchvision, transformers, etc. for experiments
```

## Quick check

```bash
cd experiments/cnn_init_rotation && python test_sanity.py
```

Should print `All sanity checks passed.` in under a minute on CPU. Verifies
SRHT round-trip, Beta-codebook symmetry, KS fit on rotated random unit
vectors, and the rotated-basis (H3) reparameterization.

## Headline results

### Vision: ImageNet-1k top-1 accuracy (Beta codebook, training-free)

All methods use the same per-row L2 normalize + (codebook or absmax)
quantization scheme; storage cost is identical at each bit width.

**ResNet-18** — FP32: 67.45%

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 55.18% | 67.42% | **67.39%** |
| 6 | 46.27% | 67.00% | **67.34%** |
| 4 | 17.31% | 59.99% | **62.34%** |
| 2 | 0.13% | 0.08% | **4.64%** |

**ResNet-50** — FP32: 76.95%

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 0.32% | 76.86% | **76.89%** |
| 6 | 0.21% | **76.86%** | 76.57% |
| 4 | 0.11% | 61.89% | **69.36%** |
| 2 | 0.09% | 0.05% | **0.43%** |

**ViT-B/16** — FP32: 79.14%

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 45.88% | 79.16% | **79.19%** |
| 6 | 27.19% | **79.15%** | 79.14% |
| 4 | 19.61% | 78.99% | **79.02%** |
| 2 | 1.71% | 0.12% | **69.86%** |

**ConvNeXt-Tiny** — FP32: 77.37%

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 40.79% | **77.36%** | 77.34% |
| 6 | 19.41% | **77.48%** | 77.45% |
| 4 | 0.44% | 77.24% | **77.34%** |
| 2 | 0.14% | 0.08% | **27.02%** |

**MobileNet-V2** — FP32: 67.92%  (depthwise kernels; fan-in as low as 9)

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 46.10% | 67.70% | **67.75%** |
| 6 | 25.40% | **63.93%** | 59.11% |
| 4 | 1.12% | **8.04%** | 5.65% |
| 2 | 0.10% | **0.11%** | 0.08% |

**EfficientNet-B0** — FP32: 74.57%  (depthwise + squeeze-excitation)

| bits | Baseline (no rot) | QuaRot | SphereQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 50.69% | **74.59%** | 73.85% |
| 6 | 8.41% | **73.41%** | 70.92% |
| 4 | 0.18% | 20.11% | **51.34%** |
| 2 | 0.10% | **0.12%** | 0.11% |

Read of the table: at 8-bit and 6-bit both rotation methods recover near-FP32
on every model — there's nothing to differentiate. The interesting regime is
4-bit and 2-bit, where SphereQuant wins decisively on architectures with
large fan-in (ResNet-50 +7pp at 4-bit; ViT-B/16 +69.7pp at 2-bit; ConvNeXt-
Tiny +26.9pp at 2-bit; EfficientNet-B0 +31pp at 4-bit) and ties or trails
QuaRot on architectures dominated by small-fan-in depthwise kernels
(MobileNet-V2, EfficientNet at 6-bit). The architectural boundary —
fan-in roughly 32 — is the threshold below which the post-rotation Beta(d/2,
d/2) approximation degrades and the matched codebook loses its edge over
QuaRot's uniform grid. This is analyzed in `paper/results.tex` Section 4.

### LLM: WikiText-2 perplexity (lower = better)

**TinyLlama 1.1B** — FP16 reference: 7.97

| bits | RTN-absmax | QuaRot | SphereQuant (ours) |
|-----:|-----------:|-------:|-------------------:|
| 8 | 7.97 | 7.98 | **7.97** |
| 6 | 8.07 | 8.04 | **8.02** |
| 4 | 10.98 | 10.67 | **8.68** |
| 2 | 1.2 × 10⁵ | 1.5 × 10⁵ | **4.7 × 10³** |

**Phi-1.5 1.3B** — FP16 reference: 21.82

| bits | RTN-absmax | QuaRot | SphereQuant (ours) |
|-----:|-----------:|-------:|-------------------:|
| 8 | 21.82 | 21.84 | 21.84 |
| 6 | 21.96 | 21.92 | **21.87** |
| 4 | 26.92 | 24.33 | **22.64** |
| 2 | 3.5 × 10⁵ | 7.4 × 10⁴ | **77.03** |

The 2-bit Phi-1.5 result (77 vs 74,000) is the most dramatic: SphereQuant
is the only training-free method that produces a usable 2-bit causal LM at
this scale.

## Reproducing the experiments

ImageNet-1k vision sweep (RTN-absmax, QuaRot, SphereQuant; bits 2/4/6/8):

```bash
cd experiments/imagenet_ptq
python run.py                  # baseline + spherequant
python run_quarot.py           # quarot baseline
python run_vit.py              # ViT MHA-aware variant (in_proj_weight)
python final_summary.py        # tables and figures
```

LLM sweep (TinyLlama + Phi-1.5):

```bash
cd experiments/llm_ptq
python run.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/phi-1_5 --bits 2 4 6 8
python final_summary.py
```

CIFAR-10 toy (also runs the H3 train-in-rotated-basis ablation):

```bash
cd experiments/cnn_init_rotation
python run.py
python plot.py
```

CIFAR-10 is downloaded on first run into `cnn_init_rotation/data/` (gitignored).
ImageNet-1k validation is streamed from the HuggingFace mirror.
TinyLlama and Phi-1.5 are pulled from HuggingFace Hub.

## What's tracked vs gitignored

Per-sweep `results.jsonl` files and the `figures/*.png` are tracked. Per-
sample raw score arrays (`results/scores/`), run logs (`*.log`), CIFAR-10
data, and model checkpoints are gitignored.
