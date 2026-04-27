"""Visualize per-row L2-normalized coordinate distributions before vs after
SphereQuant's rotation, for representative layers spanning the fan-in
range of each ImageNet model. Overlays Beta(d/2, d/2) for reference.

Output: figures/fig_rotation_distributions.png  (6 models x 3 layers)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist

from spherequant.rotation_utils import apply_rotation, make_rotation
from models import load_pretrained


THIS = Path(__file__).resolve().parent
FIG_DIR = THIS / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODELS = [
    ("resnet18", "ResNet-18"),
    ("resnet50", "ResNet-50"),
    ("vit_b_16", "ViT-B/16"),
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("mobilenet_v2", "MobileNet-V2"),
    ("efficientnet_b0", "EfficientNet-B0"),
]


def per_row_l2_normalize(W: np.ndarray) -> np.ndarray:
    """W shape (N, d) -> L2-normalized rows, return flattened coordinates."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (W / norms).ravel().astype(np.float32)


def flatten_weight(m: nn.Module) -> np.ndarray | None:
    """Return (N, d) numpy weight, or None if not a quantizable layer."""
    if isinstance(m, nn.Conv2d):
        W = m.weight.detach().cpu().numpy()
        return W.reshape(W.shape[0], -1)
    if isinstance(m, nn.Linear):
        return m.weight.detach().cpu().numpy()
    return None


def collect_layers(model: nn.Module):
    """Return list of (name, fan_in, weight_matrix)."""
    out = []
    for name, m in model.named_modules():
        W = flatten_weight(m)
        if W is None:
            continue
        d = W.shape[1]
        if d < 2:  # skip degenerate
            continue
        out.append((name, d, W))
    return out


def pick_three(layers):
    """Pick layers spanning the fan-in range. Floor at d>=9 so per-row
    histograms are not pathologically noisy from 4-coord SE bottlenecks."""
    by_d = sorted([l for l in layers if l[1] >= 9], key=lambda x: x[1])
    n = len(by_d)
    return [by_d[0], by_d[n // 2], by_d[-1]]


def main():
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(13, 2.4 * len(MODELS)),
                             sharex=True)
    bins = np.linspace(-0.5, 0.5, 80)
    x_pdf = np.linspace(-0.999, 0.999, 400)

    for row_i, (mid, mname) in enumerate(MODELS):
        print(f"loading {mid} ...")
        model = load_pretrained(mid)
        model.eval()
        layers = collect_layers(model)
        picks = pick_three(layers)
        del model

        for col_j, (lname, d, W) in enumerate(picks):
            ax = axes[row_i, col_j]

            # Pre-rotation: per-row L2-normalize, flatten
            pre = per_row_l2_normalize(W)

            # Post-rotation: rotate fan-in axis, then L2-normalize
            rot = make_rotation(d, seed=0, rotation_type="srht")
            W_rot = apply_rotation(W.astype(np.float32), rot)
            post = per_row_l2_normalize(W_rot)

            # Adapt bin range to data scale (Beta gets sharp for large d)
            scale = max(np.abs(post).max(), np.abs(pre).max(), 0.05)
            local_bins = np.linspace(-1.05 * scale, 1.05 * scale, 80)

            ax.hist(pre, bins=local_bins, density=True, alpha=0.55,
                    color="#1f77b4", label="pre-rotation")
            ax.hist(post, bins=local_bins, density=True, alpha=0.55,
                    color="#ff7f0e", label="post-rotation")

            # Beta(d/2, d/2) PDF on [-1, 1], scaled to per-row norm
            a = d / 2.0
            x01 = (x_pdf + 1) / 2
            pdf = beta_dist.pdf(x01, a, a) / 2.0  # transform from [0,1] to [-1,1]
            ax.plot(x_pdf, pdf, "r-", lw=1.3, label=f"Beta({d/2:.1f},{d/2:.1f})")

            ax.set_xlim(-1.05 * scale, 1.05 * scale)
            ax.set_yticks([])
            ax.set_title(f"{mname}\n{lname}  (d={d})", fontsize=8.5)
            if row_i == 0 and col_j == 2:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(
        "Per-row L2-normalized weight coordinates: pre vs post SphereQuant rotation\n"
        "(red overlay: Beta(d/2, d/2) — the predicted post-rotation density)",
        fontsize=10, y=0.997,
    )
    fig.text(0.5, 0.005,
             "Read left to right within each row: as fan-in d grows, Beta sharpens and post-rotation locks onto it. "
             "The leftmost cells (small d) show the architectural-boundary failure: Beta is wide, "
             "the matched codebook is nearly uniform, no advantage over QuaRot.",
             ha="center", fontsize=8.5, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    out = FIG_DIR / "fig_rotation_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
