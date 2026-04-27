"""Sanity checks for the CNN init-rotation experiment.

Runs in under a minute on CPU. Exits nonzero on any failure so this can also
be used as a smoke test before the real sweep.

Checks:
  1. Orthogonality and round-trip of make_rotation (SRHT and dense).
  2. Materialized rotation matrix matches apply_rotation.
  3. RotatedConv2d / RotatedLinear forward match nn.Conv2d / nn.Linear
     with the equivalent weight W = U @ M.T.
  4. Uniform quantizer at 2 bits produces exactly 4 unique centroids.
  5. Beta codebook boundaries symmetric around 0.
  6. Seed reproducibility of the baseline CNN3 init.
  7. Beta(d/2, d/2) KS test runs without error on random rotated coords.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Put experiment dir on path so intra-package imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CNN3  # noqa: E402
from rotated_conv import RotatedConv2d, RotatedLinear  # noqa: E402
from spherequant.rotation_utils import (  # noqa: E402
    make_rotation,
    apply_rotation,
    materialize_rotation_matrix,
    build_torch_rotation,
    torch_apply_rotation,
    torch_inverse_rotation,
    SRHTRotation,
)
from spherequant.ptq import (  # noqa: E402
    uniform_codebook,
    beta_codebook,
    per_row_quantize,
)


FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# --- 1. Rotation correctness -------------------------------------------------

def test_rotation_roundtrip():
    print("1. Rotation correctness (SRHT + dense)")
    rng = np.random.RandomState(42)
    for d, expected_type in [(128, "srht"), (27, "dense"), (288, "dense")]:
        rot = make_rotation(d, seed=0, rotation_type="srht")
        kind = "SRHT" if isinstance(rot, SRHTRotation) else "dense"
        X = rng.randn(50, d).astype(np.float32)
        X_rot = apply_rotation(X, rot)
        # norm preservation
        n_before = np.linalg.norm(X, axis=1)
        n_after = np.linalg.norm(X_rot, axis=1)
        check(f"  d={d} ({kind}) norm preserved",
              np.allclose(n_before, n_after, atol=1e-4),
              f"max dev {np.max(np.abs(n_before - n_after)):.2e}")
        # round-trip
        M = materialize_rotation_matrix(rot)
        X_rot_manual = X @ M
        check(f"  d={d} ({kind}) materialized matrix matches apply_rotation",
              np.allclose(X_rot_manual, X_rot, atol=1e-4))
        # inverse
        X_back = X_rot @ M.T
        check(f"  d={d} ({kind}) M M.T == I (round-trip)",
              np.allclose(X_back, X, atol=1e-4),
              f"max dev {np.max(np.abs(X_back - X)):.2e}")


# --- 2. H3 plumbing: RotatedConv2d matches reference -------------------------

def test_rotated_conv_equivalence():
    print("2. RotatedConv2d / RotatedLinear forward == nn.Conv2d / nn.Linear with W=U@M.T")
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8)
    rc = RotatedConv2d(3, 16, 3, seed=7, padding=1)
    W_eff = rc.effective_weight().detach()
    ref = nn.Conv2d(3, 16, 3, padding=1, bias=True)
    with torch.no_grad():
        ref.weight.copy_(W_eff)
        ref.bias.copy_(rc.bias)
    y_rc = rc(x)
    y_ref = ref(x)
    check("  conv forward matches", torch.allclose(y_rc, y_ref, atol=1e-5),
          f"max dev {(y_rc - y_ref).abs().max().item():.2e}")

    # Linear
    rl = RotatedLinear(64, 10, seed=7)
    W_eff_l = rl.effective_weight().detach()
    ref_l = nn.Linear(64, 10, bias=True)
    with torch.no_grad():
        ref_l.weight.copy_(W_eff_l)
        ref_l.bias.copy_(rl.bias)
    xl = torch.randn(5, 64)
    check("  linear forward matches",
          torch.allclose(rl(xl), ref_l(xl), atol=1e-5))


# --- 3. Codebook sanity ------------------------------------------------------

def test_codebooks():
    print("3. Codebook sanity")
    b, c = uniform_codebook(2)
    check("  uniform 2-bit has 4 centroids", len(c) == 4)
    check("  uniform 2-bit symmetric around 0", np.allclose(c, -c[::-1], atol=1e-6))

    b, c = beta_codebook(d=128, bits=4)
    check("  beta 4-bit has 16 centroids", len(c) == 16)
    check("  beta centroids symmetric around 0", np.allclose(c, -c[::-1], atol=1e-3),
          f"max asym {np.max(np.abs(c + c[::-1])):.2e}")


# --- 4. Per-row quantize MSE decreases monotonically with bits ---------------

def test_mse_monotone():
    print("4. per_row_quantize MSE decreases with bits")
    rng = np.random.RandomState(0)
    W = rng.randn(64, 128).astype(np.float32)
    prev_mse = None
    for bits in [2, 4, 6, 8]:
        _, stats = per_row_quantize(W, bits, "uniform")
        if prev_mse is not None:
            check(f"  bits {bits} MSE ({stats['mse']:.4e}) < prev ({prev_mse:.4e})",
                  stats["mse"] < prev_mse)
        prev_mse = stats["mse"]


# --- 5. Seed reproducibility -------------------------------------------------

def test_seed_repro():
    print("5. Seed reproducibility of baseline CNN3 init")
    torch.manual_seed(0)
    m1 = CNN3()
    w1 = m1.classifier.weight.detach().clone()
    torch.manual_seed(0)
    m2 = CNN3()
    w2 = m2.classifier.weight.detach().clone()
    check("  two CNN3 inits with same seed produce identical FC weights",
          torch.equal(w1, w2))


# --- 6. KS test runs without error ------------------------------------------

def test_ks_runs():
    print("6. Beta-KS test runs on rotated random unit vectors")
    from spherequant.rotation_utils import beta_ks_test
    rng = np.random.RandomState(0)
    d = 128
    X = rng.randn(1000, d).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    rot = make_rotation(d, 1, "srht")
    X_rot = apply_rotation(X, rot)
    res = beta_ks_test(X_rot.ravel(), d=d)
    check("  KS D statistic is finite",
          np.isfinite(res["D_statistic"]),
          f"D={res['D_statistic']:.4f}, fit={res['interpretation']}")
    check("  rotated random unit vectors fit Beta well (D < 0.05)",
          res["D_statistic"] < 0.05,
          f"D={res['D_statistic']:.4f}")


# --- 7. Torch rotation preserves inner products ------------------------------

def test_torch_rotation():
    print("7. Torch rotation preserves inner products")
    torch.manual_seed(0)
    d = 128
    M = build_torch_rotation(d, seed=1, rotation_type="srht")
    X = torch.randn(50, d)
    X_rot = torch_apply_rotation(X, M)
    check("  norms preserved (torch)",
          torch.allclose(X.norm(dim=1), X_rot.norm(dim=1), atol=1e-4))
    X_back = torch_inverse_rotation(X_rot, M)
    check("  round-trip (torch)",
          torch.allclose(X_back, X, atol=1e-4),
          f"max dev {(X_back - X).abs().max().item():.2e}")


def main():
    print("Running sanity checks for experiments/cnn_init_rotation/\n")
    test_rotation_roundtrip()
    test_rotated_conv_equivalence()
    test_codebooks()
    test_mse_monotone()
    test_seed_repro()
    test_ks_runs()
    test_torch_rotation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
