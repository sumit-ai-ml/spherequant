"""HuggingFace causal-LM loader + size accounting for weight-only PTQ."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, dtype=torch.float16):
    """Load an HF causal-LM in fp16/bf16 for evaluation. Stays on CPU initially;
    caller moves to GPU as needed."""
    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    return model, tok


def count_quantizable_weights(model: nn.Module) -> tuple[int, int, int]:
    """Return (quantizable_weights, non_quantizable_params, total_rows).

    We quantize every nn.Linear weight. We skip:
    - Embedding tables (often tied to lm_head, complex)
    - LayerNorm / RMSNorm weights and biases
    - nn.Linear bias (stays FP32, small)

    For simplicity here we include lm_head since it's an nn.Linear. Some papers
    keep it FP32; we quantize to keep the budget honest.
    """
    n_w, n_rows = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n_w += m.weight.numel()
            n_rows += m.weight.shape[0]
    total_params = sum(p.numel() for p in model.parameters())
    return n_w, total_params - n_w, n_rows


def model_size_at_bits(model: nn.Module, bits: int | None) -> dict:
    n_w, n_other, n_rows = count_quantizable_weights(model)
    total_params = n_w + n_other
    fp16_bytes = total_params * 2  # these models ship in fp16
    fp32_bytes = total_params * 4  # reference

    result = {
        "n_weights_quant": n_w,
        "n_params_total": total_params,
        "n_rows": n_rows,
        "fp16_mb": fp16_bytes / (1024 ** 2),
        "fp32_mb": fp32_bytes / (1024 ** 2),
    }
    if bits is not None:
        code_bytes = n_w * bits / 8
        norm_bytes = n_rows * 4  # per-row scale/norm in FP32
        other_bytes = n_other * 2  # BN/LN etc stay in fp16
        quant_bytes = code_bytes + norm_bytes + other_bytes
        result["quantized_mb"] = quant_bytes / (1024 ** 2)
        # Compression reported vs fp16 (the deploy baseline for LLMs)
        result["ratio_vs_fp16"] = fp16_bytes / quant_bytes
        # Also vs fp32 for apples-to-apples with the ViT/CNN tables
        result["ratio_vs_fp32"] = fp32_bytes / quant_bytes
    return result
