# LLM Post-Training Quantization

Minimum viable LLM sweep to validate that our rotation + Beta codebook method
transfers from vision models to causal language models.

## Scope

Weight-only PTQ on:
- **TinyLlama 1.1B** (fits comfortably on 6 GB VRAM)
- **Phi-2 2.7B** (tight on 6 GB VRAM — may need fp16 eval)

Methods:
- **RTN baseline** (per-channel symmetric absmax uniform, no rotation)
- **QuaRot-RTN** (rotation + per-channel absmax uniform)
- **SphereQuant + Beta (ours)** (rotation + per-row L2 normalize + Beta codebook)

Bits: 2, 4, 6, 8
Eval: WikiText-2 perplexity (standard LLM quantization benchmark)

All methods are **training-free** (no calibration data).

## How to run

```bash
python run.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits 2 4 6 8
python final_summary.py
```

Outputs: `results/results.jsonl`, `figures/*.png`.

## Architecture compatibility

LLMs use separate `nn.Linear` modules for Q, K, V, O projections and MLP
up/gate/down projections (unlike torchvision ViT's `nn.MultiheadAttention`
which hides weights in `in_proj_weight`). Our existing `ptq.py` handles
`nn.Linear` directly — no MHA hack needed.

Typical fan-ins for TinyLlama:
- `q_proj`, `o_proj`: 2048 (power of 2, SRHT applies)
- `k_proj`, `v_proj`: 2048 (GQA → output 256)
- `gate_proj`, `up_proj`: 2048 (power of 2)
- `down_proj`: 5632 (non-power-of-2, dense orthogonal rotation)

## What counts as success

If 4-bit SphereQuant+Beta perplexity on WikiText-2 is within ~10% of FP16 perplexity
(and comparable to QuaRot), the method generalizes. If 2-bit SphereQuant+Beta is
meaningfully usable (perplexity < 2× FP16), that's a headline finding.

Reference numbers from the QuaRot paper for Llama-2 7B (not our model, but
comparable): FP16 perplexity ≈ 5.47, QuaRot+RTN 4-bit ≈ 5.72.
