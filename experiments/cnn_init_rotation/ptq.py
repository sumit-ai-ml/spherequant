"""Post-training quantization evaluators.

Three quantization paths, each applied to trained CNN3 or RotatedCNN3 weights:

1. BASELINE: quantize W directly (no rotation). Per-row L2 normalize,
   quantize the unit-sphere rows with uniform or Beta codebook, store norms.
   This is the "what does TurboQuant-style quantization look like WITHOUT the
   rotation step" control. Not the same as weight-magnitude quantization.

2. H2 (rotate-then-quantize): take a baseline-trained CNN3, rotate each
   layer's flattened weight on the fan-in axis (W_flat @ M), then quantize
   the rotated rows. TurboQuant's original pipeline applied to CNN weights.

3. H3 (quantize-U): take a RotatedCNN3 (H3-trained). U is already the "rotated
   coordinates." Quantize U, reconstruct W = U_q @ M.T. This tests whether
   training in the rotated basis produces a U whose distribution is more
   quantizable than baseline W or H2's post-hoc-rotated W.

All paths share the same codebooks (uniform grid on [-1,1] or Beta(d/2,d/2))
and the same per-row normalization, so storage cost is identical across
variants at a given bit budget.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist
import torch
import torch.nn as nn

from rotation_utils import (
    build_torch_rotation,
    apply_rotation,
    SRHTRotation,
    make_rotation,
)

# Only import KS test from parent; reimplement the Beta codebook locally
# because the parent uses np.trapezoid (numpy >= 2.0) and our env is older.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import beta_ks_test  # noqa: E402


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

    Same algorithm as TurboQuantMSE._build_codebook in the parent repo, but
    uses np.trapz instead of np.trapezoid for numpy <2.0 compatibility.
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
    # Shift from [0, 1] to [-1, 1]
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

    Storage cost per row: d * bits (for the codes) + 4 bytes (FP32 norm).

    Args:
        W: (N, d) FP32 matrix. Rows will be L2-normalized for codebook lookup.
        bits: 2, 4, 6, 8, ...
        codebook: "uniform" or "beta"

    Returns:
        W_reconstructed: (N, d) float32 after round-trip
        stats: dict with ks_D (KS vs Beta(d/2,d/2) on normalized rows),
               mse (||W - W_rec||^2 / W.size), n_levels, codebook
    """
    assert W.ndim == 2, f"expected (N, d) matrix, got {W.shape}"
    N, d = W.shape
    W = W.astype(np.float32)

    # Row-wise L2 norm (store for reconstruction)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    W_unit = W / norms_safe  # rows on unit sphere, each coord in [-1, 1]

    if codebook == "uniform":
        boundaries, centroids = uniform_codebook(bits)
    elif codebook == "beta":
        boundaries, centroids = beta_codebook(d, bits)
    else:
        raise ValueError(f"unknown codebook: {codebook}")

    # KS statistic against Beta(d/2,d/2) on the normalized-row coordinates
    ks = beta_ks_test(W_unit.ravel(), d=d)

    W_unit_q = quantize_with_codebook(W_unit, boundaries, centroids)
    W_reconstructed = W_unit_q * norms  # multiply back by original norms

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
# Per-layer quantization dispatch for baseline/H2/H3
# --------------------------------------------------------------------------

@dataclass
class LayerStats:
    name: str
    d: int
    mse: float
    ks_D: float


def _flatten_conv_or_linear(m: nn.Module) -> tuple[np.ndarray, tuple, str]:
    """Return (W_flat, original_shape, kind)."""
    if isinstance(m, nn.Conv2d):
        W = m.weight.detach().cpu().numpy()
        orig_shape = W.shape  # (C_out, C_in, kH, kW)
        W_flat = W.reshape(orig_shape[0], -1)  # (C_out, C_in*kH*kW)
        return W_flat, orig_shape, "conv"
    if isinstance(m, nn.Linear):
        W = m.weight.detach().cpu().numpy()
        orig_shape = W.shape  # (out, in)
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


def quantize_model_h2(model: nn.Module, bits: int, codebook: str,
                      rotation_seed: int = 0,
                      rotation_type: str = "srht") -> tuple[nn.Module, list[LayerStats]]:
    """H2: rotate each layer's fan-in axis, then quantize. Reconstruct with M.T."""
    model_q = copy.deepcopy(model)
    stats = []
    for idx, (name, m) in enumerate(
        [(n, mod) for n, mod in model_q.named_modules()
         if isinstance(mod, (nn.Conv2d, nn.Linear))]
    ):
        W_flat, orig_shape, _ = _flatten_conv_or_linear(m)
        N, d = W_flat.shape
        # Per-layer rotation seed so different layers get different Ms
        per_layer_seed = rotation_seed * 1000 + idx + 1
        rot = make_rotation(d, per_layer_seed, rotation_type)
        # Rotate rows: U = W_flat @ M (equivalently apply_rotation in parent repo)
        U = apply_rotation(W_flat.astype(np.float32), rot)  # (N, d)
        U_rec, s = per_row_quantize(U, bits, codebook)
        # Inverse rotation: W = U_rec @ M.T
        if isinstance(rot, SRHTRotation):
            W_rec = rot.inverse(U_rec)
        else:
            # For dense: apply_rotation does X @ rot.T. Inverse is X @ rot.
            W_rec = U_rec @ rot
        _write_weight_back(m, W_rec, orig_shape)
        stats.append(LayerStats(name, d, s["mse"], s["ks_D"]))
    return model_q, stats


def quantize_model_rtn_absmax(model: nn.Module, bits: int) -> tuple[nn.Module, list[LayerStats]]:
    """Round-to-nearest with per-channel symmetric absmax quantization.

    The canonical RTN baseline used in LLM quantization papers (GPTQ, AWQ,
    QuaRot). No rotation. For each row, scale = max|x|, then quantize to
    2^(bits-1) - 1 levels symmetrically.
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
    """QuaRot-style baseline (Ashkboos et al., NeurIPS 2024) adapted for CNN weights.

    Hadamard/SRHT rotation on the fan-in axis + per-channel symmetric uniform
    quantization (absmax scale per output row). Storage cost matches H2:
    d * bits per weight code + 1 FP32 scale per row.

    Differences from our H2:
      - H2 uses per-row L2 normalize + [-1,1] codebook (uniform or Beta)
      - QuaRot uses absmax-scaled symmetric integer quant: q = round(x / s * L)
        where s = max(|x|), L = 2^(bits-1) - 1 (e.g. bits=4 -> L=7, levels=-7..+7)
    This matches how QuaRot quantizes weights in the original LLM paper
    (their weight-only setup is equivalent once activation quantization is removed).
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

        # Per-channel symmetric absmax uniform quantization
        scale = np.max(np.abs(U), axis=1, keepdims=True).astype(np.float32)
        scale = np.where(scale < 1e-12, 1.0, scale)
        levels = (1 << (bits - 1)) - 1  # e.g. bits=4 -> 7
        if bits == 1:
            levels = 1  # degenerate
        U_int = np.round(U / scale * levels).clip(-levels, levels)
        U_rec = (U_int * scale / levels).astype(np.float32)

        # Inverse rotation
        if isinstance(rot, SRHTRotation):
            W_rec = rot.inverse(U_rec)
        else:
            W_rec = U_rec @ rot

        _write_weight_back(m, W_rec, orig_shape)
        # KS against Beta(d/2, d/2) on post-rotation coordinates (for symmetry)
        U_unit = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        ks = beta_ks_test(U_unit.ravel(), d=d)
        mse = float(((W_flat - W_rec) ** 2).mean())
        stats.append(LayerStats(name, d, mse, ks["D_statistic"]))
    return model_q, stats


def quantize_model_h3(rotated_model, bits: int, codebook: str) -> tuple[nn.Module, list[LayerStats]]:
    """H3: model already stores U (rotated coords) and M (buffer).
    Quantize U directly, then the effective_weight() reconstructs W = U_q @ M.T.
    """
    model_q = copy.deepcopy(rotated_model)
    stats = []
    for name, m in model_q.rotated_layers():
        U = m.U.detach().cpu().numpy()  # (out, fan_in)
        d = U.shape[1]
        U_rec, s = per_row_quantize(U, bits, codebook)
        with torch.no_grad():
            m.U.copy_(torch.from_numpy(U_rec).to(m.U.dtype).to(m.U.device))
        stats.append(LayerStats(name, d, s["mse"], s["ks_D"]))
    return model_q, stats


# --------------------------------------------------------------------------
# Evaluation on a model: returns accuracy, softmax scores, labels
# (Scores cached so AUROC / other metrics can be recomputed without re-running.)
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
