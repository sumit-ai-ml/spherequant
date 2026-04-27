"""Post-training quantization evaluators.

Quantization paths, all per-row-normalized so the storage cost is identical
across variants at a given bit budget (b bits per code + 1 FP32 norm per row):

1. BASELINE: quantize W directly (no rotation). L2-normalize each row,
   quantize the unit-sphere coordinates with the chosen codebook, store norms.
   Control for "what does codebook quantization look like WITHOUT rotation".

2. SPHEREQUANT (rotate-then-quantize, ours): rotate each layer's flattened
   weight on the fan-in axis (W_flat @ M), L2-normalize the rotated rows,
   quantize, then reconstruct via the inverse rotation.

3. QUAROT-style baseline (Ashkboos et al., NeurIPS 2024): rotation + per-row
   absmax-scaled symmetric uniform quantization. Same rotation as SphereQuant,
   but quantizes with a uniform integer grid instead of a Beta-matched codebook.

4. RTN-ABSMAX: round-to-nearest with per-channel symmetric absmax scaling.
   No rotation. The canonical baseline used in GPTQ / AWQ / QuaRot papers.

5. H3 (train in rotated basis, ablation): the model already stores U
   (rotated coordinates) and a buffer M. Quantize U directly; the layer's
   forward pass reconstructs W = U_q @ M.T. Tests whether training in the
   rotated basis produces a U whose distribution is more quantizable than
   post-hoc rotation of a baseline-trained W.

All paths share the same codebooks (uniform grid on [-1, 1] or
Beta(d/2, d/2) Lloyd-Max) and the same per-row normalization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist

from spherequant.rotation_utils import (
    SRHTRotation,
    apply_rotation,
    beta_ks_test,
    make_rotation,
)


# --------------------------------------------------------------------------
# Codebooks on the unit-sphere-normalized coordinate range [-1, 1]
# --------------------------------------------------------------------------

def uniform_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid on [-1, 1]. Returns (boundaries, centroids)."""
    n_levels = 2 ** bits
    boundaries = np.linspace(-1.0, 1.0, n_levels + 1, dtype=np.float32)
    centroids = ((boundaries[:-1] + boundaries[1:]) / 2).astype(np.float32)
    return boundaries, centroids


def beta_codebook(d: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Beta(d/2, d/2) Lloyd-Max codebook on [-1, 1].

    Iterates Lloyd-Max in [0, 1] then shifts to [-1, 1]. Uses np.trapz for
    compatibility with numpy < 2.0.
    """
    n_levels = 2 ** bits
    a = b = d / 2.0
    boundaries = beta_dist.ppf(np.linspace(0, 1, n_levels + 1), a, b)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    for _ in range(100):
        centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            x = np.linspace(lo + 1e-10, hi - 1e-10, 1000)
            pdf = beta_dist.pdf(x, a, b)
            mass = np.trapz(pdf, x)
            if mass > 1e-15:
                centroids[i] = np.trapz(x * pdf, x) / mass
            else:
                centroids[i] = (lo + hi) / 2
        new_boundaries = np.zeros(n_levels + 1)
        new_boundaries[0] = 0.0
        new_boundaries[-1] = 1.0
        for i in range(1, n_levels):
            new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2
        if np.allclose(boundaries, new_boundaries, atol=1e-10):
            break
        boundaries = new_boundaries
    boundaries = (boundaries * 2 - 1).astype(np.float32)
    centroids = (centroids * 2 - 1).astype(np.float32)
    return boundaries, centroids


def quantize_with_codebook(x: np.ndarray, boundaries: np.ndarray,
                           centroids: np.ndarray) -> np.ndarray:
    """Map x (any shape) into nearest centroid via digitize on interior boundaries."""
    idx = np.digitize(x, boundaries[1:-1]).astype(np.int64)
    return centroids[idx]


# --------------------------------------------------------------------------
# Per-row-normalized quantization of a matrix (rows are filters, d = fan-in)
# --------------------------------------------------------------------------

def per_row_quantize(W: np.ndarray, bits: int, codebook: str) -> tuple[np.ndarray, dict]:
    """Quantize each row of W independently after L2-normalizing.

    Storage cost per row: d * bits (codes) + 4 bytes (FP32 norm).

    Args:
        W: (N, d) FP32 matrix.
        bits: 2, 4, 6, 8, ...
        codebook: "uniform" or "beta".

    Returns:
        (W_reconstructed, stats) where stats includes ks_D, mse, n_levels.
    """
    assert W.ndim == 2, f"expected (N, d) matrix, got {W.shape}"
    N, d = W.shape
    W = W.astype(np.float32)

    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    W_unit = W / norms_safe

    if codebook == "uniform":
        boundaries, centroids = uniform_codebook(bits)
    elif codebook == "beta":
        boundaries, centroids = beta_codebook(d, bits)
    else:
        raise ValueError(f"unknown codebook: {codebook}")

    ks = beta_ks_test(W_unit.ravel(), d=d)
    W_unit_q = quantize_with_codebook(W_unit, boundaries, centroids)
    W_reconstructed = W_unit_q * norms

    mse = float(((W - W_reconstructed) ** 2).mean())
    return W_reconstructed.astype(np.float32), {
        "ks_D": ks["D_statistic"],
        "ks_fit": ks["interpretation"],
        "mse": mse,
        "n_levels": 2 ** bits,
        "codebook": codebook,
        "n_rows": N,
        "d": d,
    }


# --------------------------------------------------------------------------
# Per-layer quantization dispatch
# --------------------------------------------------------------------------

@dataclass
class LayerStats:
    name: str
    d: int
    mse: float
    ks_D: float


def _flatten_conv_or_linear(m: nn.Module) -> tuple[np.ndarray, tuple, str]:
    if isinstance(m, nn.Conv2d):
        W = m.weight.detach().cpu().numpy()
        orig_shape = W.shape  # (C_out, C_in, kH, kW)
        W_flat = W.reshape(orig_shape[0], -1)
        return W_flat, orig_shape, "conv"
    if isinstance(m, nn.Linear):
        W = m.weight.detach().cpu().numpy()
        orig_shape = W.shape
        return W.copy(), orig_shape, "linear"
    raise TypeError(f"unsupported module: {type(m).__name__}")


def _write_weight_back(m: nn.Module, W_flat: np.ndarray, orig_shape: tuple):
    W = W_flat.reshape(orig_shape)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(W).to(m.weight.dtype).to(m.weight.device))


def quantize_model_baseline(model: nn.Module, bits: int, codebook: str) -> tuple[nn.Module, list[LayerStats]]:
    """Quantize each layer's weight directly (no rotation)."""
    model_q = copy.deepcopy(model)
    stats = []
    for name, m in model_q.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        W_flat, orig_shape, _ = _flatten_conv_or_linear(m)
        W_rec, s = per_row_quantize(W_flat, bits, codebook)
        _write_weight_back(m, W_rec, orig_shape)
        stats.append(LayerStats(name, s["d"], s["mse"], s["ks_D"]))
    return model_q, stats


def quantize_model_spherequant(model: nn.Module, bits: int, codebook: str,
                               rotation_seed: int = 0,
                               rotation_type: str = "srht") -> tuple[nn.Module, list[LayerStats]]:
    """SphereQuant: rotate each layer's fan-in axis, then per-row-normalize and
    quantize the rotated coordinates with the (Beta or uniform) codebook.
    Reconstruct via the inverse rotation.
    """
    model_q = copy.deepcopy(model)
    stats = []
    for idx, (name, m) in enumerate(
        [(n, mod) for n, mod in model_q.named_modules()
         if isinstance(mod, (nn.Conv2d, nn.Linear))]
    ):
        W_flat, orig_shape, _ = _flatten_conv_or_linear(m)
        N, d = W_flat.shape
        per_layer_seed = rotation_seed * 1000 + idx + 1
        rot = make_rotation(d, per_layer_seed, rotation_type)
        U = apply_rotation(W_flat.astype(np.float32), rot)
        U_rec, s = per_row_quantize(U, bits, codebook)
        if isinstance(rot, SRHTRotation):
            W_rec = rot.inverse(U_rec)
        else:
            # apply_rotation does X @ rot.T; inverse is X @ rot.
            W_rec = U_rec @ rot
        _write_weight_back(m, W_rec, orig_shape)
        stats.append(LayerStats(name, d, s["mse"], s["ks_D"]))
    return model_q, stats


def quantize_model_rtn_absmax(model: nn.Module, bits: int) -> tuple[nn.Module, list[LayerStats]]:
    """Round-to-nearest with per-channel symmetric absmax quantization.

    Canonical RTN baseline used in LLM quantization papers (GPTQ, AWQ, QuaRot).
    No rotation. For each row: scale = max|x|, quantize to 2^(bits-1) - 1 levels.
    """
    model_q = copy.deepcopy(model)
    stats = []
    for name, m in model_q.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        W_flat, orig_shape, _ = _flatten_conv_or_linear(m)
        N, d = W_flat.shape
        scale = np.max(np.abs(W_flat), axis=1, keepdims=True).astype(np.float32)
        scale = np.where(scale < 1e-12, 1.0, scale)
        levels = (1 << (bits - 1)) - 1
        if bits == 1:
            levels = 1
        W_int = np.round(W_flat / scale * levels).clip(-levels, levels)
        W_rec = (W_int * scale / levels).astype(np.float32)
        _write_weight_back(m, W_rec, orig_shape)
        mse = float(((W_flat - W_rec) ** 2).mean())
        stats.append(LayerStats(name, d, mse, 0.0))
    return model_q, stats


def quantize_model_quarot(model: nn.Module, bits: int,
                          rotation_seed: int = 0,
                          rotation_type: str = "srht") -> tuple[nn.Module, list[LayerStats]]:
    """QuaRot-style baseline (Ashkboos et al., NeurIPS 2024) on CNN/MLP weights.

    Hadamard / SRHT rotation on the fan-in axis + per-channel symmetric uniform
    quantization (absmax scale per output row). Same storage cost as SphereQuant:
    d * bits per code + 1 FP32 scale per row.

    Difference from SphereQuant: SphereQuant uses per-row L2 normalize + a
    Beta-matched codebook on [-1, 1]. QuaRot uses absmax-scaled symmetric
    integer quant on a uniform grid.
    """
    model_q = copy.deepcopy(model)
    stats = []
    for idx, (name, m) in enumerate(
        [(n, mod) for n, mod in model_q.named_modules()
         if isinstance(mod, (nn.Conv2d, nn.Linear))]
    ):
        W_flat, orig_shape, _ = _flatten_conv_or_linear(m)
        N, d = W_flat.shape
        per_layer_seed = rotation_seed * 1000 + idx + 1
        rot = make_rotation(d, per_layer_seed, rotation_type)
        U = apply_rotation(W_flat.astype(np.float32), rot)

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

        _write_weight_back(m, W_rec, orig_shape)
        U_unit = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        ks = beta_ks_test(U_unit.ravel(), d=d)
        mse = float(((W_flat - W_rec) ** 2).mean())
        stats.append(LayerStats(name, d, mse, ks["D_statistic"]))
    return model_q, stats


def quantize_model_h3(rotated_model, bits: int, codebook: str) -> tuple[nn.Module, list[LayerStats]]:
    """H3 ablation: model already stores U (rotated coords) and M (buffer).
    Quantize U directly; the layer's effective_weight() reconstructs W = U_q @ M.T.

    H3 is a separate ablation (train in rotated basis), not the headline method.
    Kept here because the cnn_init_rotation toy uses the same dispatch surface.
    """
    model_q = copy.deepcopy(rotated_model)
    stats = []
    for name, m in model_q.rotated_layers():
        U = m.U.detach().cpu().numpy()
        d = U.shape[1]
        U_rec, s = per_row_quantize(U, bits, codebook)
        with torch.no_grad():
            m.U.copy_(torch.from_numpy(U_rec).to(m.U.dtype).to(m.U.device))
        stats.append(LayerStats(name, d, s["mse"], s["ks_D"]))
    return model_q, stats


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model: nn.Module, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def eval_with_scores(model: nn.Module, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (accuracy, scores of shape (N, C), labels of shape (N,)).
    Scores are softmax probabilities, suitable for AUROC."""
    model.eval()
    all_scores = []
    all_labels = []
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_scores.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    return correct / total, scores, labels


def compute_auroc_macro(scores: np.ndarray, labels: np.ndarray) -> float:
    """Macro-average one-vs-rest AUROC over C classes."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores, multi_class="ovr", average="macro"))
