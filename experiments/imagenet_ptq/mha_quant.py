"""Supplementary quantization for nn.MultiheadAttention.in_proj_weight.

torchvision's ViT (and many other Transformer impls) use nn.MultiheadAttention,
which holds the Q+K+V projection as a single Parameter `in_proj_weight` of shape
(3*embed_dim, embed_dim). This Parameter is NOT inside a Conv2d or Linear module,
so the standard ptq.py iteration misses it.

This module post-processes a quantized model: for every MultiheadAttention layer,
treat in_proj_weight as a (3*embed_dim, embed_dim) Linear weight and quantize it
with the same baseline / SphereQuant / QuaRot scheme used for the main model.

Call AFTER quantize_model_{baseline, sq, quarot} so the MHA weights also get
quantized rather than staying FP32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from spherequant.ptq import (  # noqa: E402
    per_row_quantize, apply_rotation, make_rotation, SRHTRotation,
)


def _iter_mha(model: nn.Module):
    """Yield (full_name, mha_module) pairs."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            yield name, mod


def has_mha(model: nn.Module) -> bool:
    return any(True for _ in _iter_mha(model))


def count_mha_params(model: nn.Module) -> int:
    n = 0
    for _, mha in _iter_mha(model):
        if mha.in_proj_weight is not None:
            n += mha.in_proj_weight.numel()
        # Also out_proj.weight is already nn.Linear, captured by main pipeline.
    return n


def _quantize_in_proj_baseline(W: torch.Tensor, bits: int, codebook: str) -> torch.Tensor:
    W_np = W.detach().cpu().numpy().astype(np.float32)
    W_rec, _ = per_row_quantize(W_np, bits, codebook)
    return torch.from_numpy(W_rec).to(W.dtype).to(W.device)


def _quantize_in_proj_spherequant(W: torch.Tensor, bits: int, codebook: str,
                         rotation_seed: int, layer_idx: int) -> torch.Tensor:
    W_np = W.detach().cpu().numpy().astype(np.float32)
    N, d = W_np.shape
    rot = make_rotation(d, rotation_seed * 1000 + 100 + layer_idx, "srht")
    U = apply_rotation(W_np, rot)
    U_rec, _ = per_row_quantize(U, bits, codebook)
    if isinstance(rot, SRHTRotation):
        W_rec = rot.inverse(U_rec)
    else:
        W_rec = U_rec @ rot
    return torch.from_numpy(W_rec.astype(np.float32)).to(W.dtype).to(W.device)


def _quantize_in_proj_quarot(W: torch.Tensor, bits: int,
                             rotation_seed: int, layer_idx: int) -> torch.Tensor:
    W_np = W.detach().cpu().numpy().astype(np.float32)
    N, d = W_np.shape
    rot = make_rotation(d, rotation_seed * 1000 + 100 + layer_idx, "srht")
    U = apply_rotation(W_np, rot)
    scale = np.max(np.abs(U), axis=1, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-12, 1.0, scale)
    levels = (1 << (bits - 1)) - 1
    if bits == 1:
        levels = 1
    U_int = np.round(U / scale * levels).clip(-levels, levels)
    U_rec = (U_int * scale / levels).astype(np.float32)
    if isinstance(rot, SRHTRotation):
        W_rec = rot.inverse(U_rec)
    else:
        W_rec = U_rec @ rot
    return torch.from_numpy(W_rec.astype(np.float32)).to(W.dtype).to(W.device)


def quantize_mha_in_place(model: nn.Module, variant: str, bits: int,
                          codebook: str = "beta", rotation_seed: int = 0):
    """Mutate model: quantize every nn.MultiheadAttention.in_proj_weight.

    variant: "baseline" | "spherequant" | "quarot"
    codebook: ignored for "quarot"; "uniform" or "beta" otherwise.
    """
    for layer_idx, (_, mha) in enumerate(_iter_mha(model)):
        if mha.in_proj_weight is None:
            continue
        W = mha.in_proj_weight  # shape (3*embed_dim, embed_dim)
        if variant == "baseline":
            W_rec = _quantize_in_proj_baseline(W, bits, codebook)
        elif variant == "spherequant":
            W_rec = _quantize_in_proj_spherequant(W, bits, codebook,
                                          rotation_seed, layer_idx)
        elif variant == "quarot":
            W_rec = _quantize_in_proj_quarot(W, bits, rotation_seed, layer_idx)
        else:
            raise ValueError(f"unknown variant {variant}")
        with torch.no_grad():
            mha.in_proj_weight.copy_(W_rec)
    return model


def model_size_with_mha(model: nn.Module, bits: int) -> dict:
    """Refined size accounting that includes MHA in_proj_weight as quantizable."""
    total_params = sum(p.numel() for p in model.parameters())
    n_w_main = 0
    n_rows = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_w_main += m.weight.numel()
            n_rows += m.weight.shape[0]
    n_w_mha = 0
    for _, mha in _iter_mha(model):
        if mha.in_proj_weight is not None:
            n_w_mha += mha.in_proj_weight.numel()
            n_rows += mha.in_proj_weight.shape[0]
    n_w_total = n_w_main + n_w_mha
    fp32_bytes = total_params * 4
    code_bytes = n_w_total * bits / 8
    norm_bytes = n_rows * 4
    other_bytes = (total_params - n_w_total) * 4
    quant_bytes = code_bytes + norm_bytes + other_bytes
    return {
        "fp32_mb": fp32_bytes / (1024 ** 2),
        "quantized_mb": quant_bytes / (1024 ** 2),
        "ratio": fp32_bytes / quant_bytes,
        "n_weights_quant": n_w_total,
        "n_weights_main": n_w_main,
        "n_weights_mha": n_w_mha,
        "n_params_total": total_params,
        "n_rows": n_rows,
    }
