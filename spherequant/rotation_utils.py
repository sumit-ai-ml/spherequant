"""Rotation primitives and Beta(d/2, d/2) statistics.

Numpy primitives:
  - SRHTRotation: structured randomized Hadamard transform, O(d log d).
    Power-of-2 d only.
  - random_orthogonal: dense Q from QR of a random Gaussian. O(d^2) memory.
  - make_rotation: dispatches to SRHT for power-of-2 d, dense otherwise.
  - apply_rotation: rotate (N, d) embeddings.
  - verify_rotation: orthogonality round-trip check.

Torch helpers:
  - build_torch_rotation: materialize the rotation as a torch (d, d) matrix.
  - torch_apply_rotation / torch_inverse_rotation: matmul wrappers.

Statistics:
  - beta_ks_test: KS statistic of rotated coords against Beta(d/2, d/2).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import hadamard
from scipy.stats import beta as beta_dist
from scipy.stats import kstest


def random_orthogonal(d: int, seed: int) -> np.ndarray:
    """Dense random orthogonal matrix via QR decomposition. O(d^2) per vector."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


class SRHTRotation:
    """Structured Randomized Hadamard Transform. O(d log d) per vector.

    Only correct for power-of-2 d. For other dimensions, use random_orthogonal()
    or make_rotation() (which auto-falls back to dense).
    """

    def __init__(self, d: int, seed: int):
        self.d = d
        if d & (d - 1) != 0:
            raise ValueError(
                f"SRHTRotation requires d to be a power of 2, got d={d}. "
                f"Use make_rotation(d, seed, 'srht') which auto-falls back to dense."
            )
        rng = np.random.RandomState(seed)
        self.signs_d = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        self.signs_s = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        self.H = hadamard(d).astype(np.float32) / np.sqrt(d)

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        rotated = embeddings * self.signs_s[None, :]
        rotated = rotated @ self.H.T
        rotated = rotated * self.signs_d[None, :]
        return rotated

    def inverse(self, rotated: np.ndarray) -> np.ndarray:
        inv = rotated * self.signs_d[None, :]
        inv = inv @ self.H.T  # H is symmetric and orthogonal
        inv = inv * self.signs_s[None, :]
        return inv


def make_rotation(d: int, seed: int, rotation_type: str = "srht"):
    """Build a rotation. SRHT for power-of-2 d, dense orthogonal otherwise.

    Returns either an SRHTRotation object or a dense orthogonal np.ndarray.
    """
    is_power_of_2 = (d & (d - 1) == 0) and d > 0
    if rotation_type == "srht" and is_power_of_2:
        return SRHTRotation(d, seed)
    return random_orthogonal(d, seed)


def apply_rotation(embeddings: np.ndarray, rotation) -> np.ndarray:
    """Apply rotation to (N, d) embeddings. Supports SRHT or dense matrix."""
    if isinstance(rotation, SRHTRotation):
        return rotation.forward(embeddings)
    return embeddings @ rotation.T


def verify_rotation(rotation, atol: float = 1e-3) -> bool:
    """Round-trip orthogonality check on a random batch of vectors."""
    d = rotation.d if isinstance(rotation, SRHTRotation) else rotation.shape[0]
    rng = np.random.RandomState(0)
    X = rng.randn(8, d).astype(np.float32)
    Y = apply_rotation(X, rotation)
    if isinstance(rotation, SRHTRotation):
        X_rec = rotation.inverse(Y)
    else:
        X_rec = Y @ rotation
    return np.allclose(X, X_rec, atol=atol)


def materialize_rotation_matrix(rot) -> np.ndarray:
    """Return the dense (d, d) matrix M such that apply_rotation(X, rot) == X @ M.

    For SRHT: M = forward(I). For dense orthogonal Q: M = Q.T.
    Materializing costs O(d^2) memory; fine for d up to a few thousand.
    """
    if isinstance(rot, SRHTRotation):
        d = rot.d
        I = np.eye(d, dtype=np.float32)
        return rot.forward(I).astype(np.float32)
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
    """Inverse rotation: returns W_rot @ M.T. M is orthogonal."""
    return W_rot @ M.T


def beta_ks_test(rotated_coords: np.ndarray, d: int) -> dict:
    """KS test of rotated coordinates against Beta(d/2, d/2).

    After rotating a unit-norm vector by a random orthogonal matrix, each
    coordinate follows Beta(d/2, d/2) on [-1, 1]. We test the empirical
    distribution against this reference. Returns D statistic, p-value, and
    a coarse fit interpretation.
    """
    shifted = (rotated_coords + 1.0) / 2.0
    shifted = np.clip(shifted, 1e-10, 1 - 1e-10)
    a = b = d / 2.0
    D, p_value = kstest(shifted, beta_dist(a, b).cdf)

    if D < 0.01:
        interpretation = "excellent_fit"
    elif D < 0.02:
        interpretation = "good_fit"
    elif D < 0.05:
        interpretation = "moderate_fit"
    else:
        interpretation = "poor_fit"

    return {
        "D_statistic": float(D),
        "p_value": float(p_value),
        "interpretation": interpretation,
        "n_samples": len(rotated_coords),
    }
