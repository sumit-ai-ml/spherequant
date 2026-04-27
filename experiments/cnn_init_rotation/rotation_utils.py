"""Torch wrappers around the parent repo's numpy rotation primitives.

We import SRHTRotation / make_rotation / apply_rotation from ../../utils.py
without modifying them, and expose:
  - materialize_rotation_matrix(rot) -> np.ndarray (d, d) dense form M
    such that apply_rotation(X, rot) == X @ M
  - torch_apply_rotation(W_torch, M_torch) -> torch.Tensor
    applies rotation through a torch matmul (supports autograd)
"""

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import SRHTRotation, make_rotation, apply_rotation  # noqa: E402


def materialize_rotation_matrix(rot) -> np.ndarray:
    """Return the dense (d, d) matrix M such that apply_rotation(X, rot) == X @ M.

    For SRHT: M = forward(I). For dense orthogonal Q (numpy): M = Q.T.
    Materializing costs O(d^2) memory; fine for d <= a few thousand.
    """
    if isinstance(rot, SRHTRotation):
        d = rot.d
        I = np.eye(d, dtype=np.float32)
        return rot.forward(I).astype(np.float32)
    # dense orthogonal returned by random_orthogonal
    return rot.T.astype(np.float32)


def build_torch_rotation(d: int, seed: int, rotation_type: str = "srht",
                         device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Build a torch (d, d) rotation matrix M suitable for X_torch @ M."""
    rot = make_rotation(d, seed, rotation_type)
    M_np = materialize_rotation_matrix(rot)
    return torch.from_numpy(M_np).to(device=device, dtype=dtype)


def torch_apply_rotation(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Rotate rows of W: returns W @ M. W is (N, d), M is (d, d)."""
    return W @ M


def torch_inverse_rotation(W_rot: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: returns W_rot @ M.T. M is orthogonal, so M^-1 == M.T."""
    return W_rot @ M.T


# Convenience re-exports so experiment code doesn't need to touch sys.path
__all__ = [
    "SRHTRotation",
    "make_rotation",
    "apply_rotation",
    "materialize_rotation_matrix",
    "build_torch_rotation",
    "torch_apply_rotation",
    "torch_inverse_rotation",
]
