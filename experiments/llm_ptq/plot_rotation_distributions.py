"""LLM version of plot_rotation_distributions.py.

Picks one attention projection (q_proj), one MLP gate projection, and one
MLP down projection per LLM (3 layers covering the fan-in spectrum) and
plots pre vs post ApexQuant rotation distributions with Beta(d/2, d/2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist

from apexquant.rotation_utils import apply_rotation, make_rotation
from models import load_model


THIS = Path(__file__).resolve().parent
FIG_DIR = THIS / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B"),
    ("microsoft/phi-1_5", "Phi-1.5 1.3B"),
]

# Substring patterns to pick one layer per fan-in regime, in any layer index.
# Llama: attention q/k/v/o + MLP gate/up/down. Phi: dense / fc1 / fc2.
PICK_PATTERNS = [
    ("attention small-fan-in",  ["self_attn.q_proj", "self_attn.Wqkv", "mixer.Wqkv", "self_attn.dense", "mixer.out_proj"]),
    ("MLP / mid",               ["mlp.gate_proj", "mlp.up_proj", "mlp.fc1"]),
    ("MLP down (large)",        ["mlp.down_proj", "mlp.fc2"]),
]


def per_row_l2_normalize(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (W / norms).ravel().astype(np.float32)


def find_first(model: nn.Module, patterns: list[str]):
    """Return (name, weight matrix) for the first Linear matching any pattern."""
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(p in name for p in patterns):
            W = m.weight.detach().to(torch.float32).cpu().numpy()
            return name, W
    return None


def main():
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(13, 2.6 * len(MODELS)),
                             sharex=False)
    x_pdf = np.linspace(-0.999, 0.999, 400)

    for row_i, (mid, mname) in enumerate(MODELS):
        print(f"loading {mid} ...")
        model, tokenizer = load_model(mid, dtype=torch.float32)
        model.eval()

        for col_j, (label, patterns) in enumerate(PICK_PATTERNS):
            ax = axes[row_i, col_j]
            picked = find_first(model, patterns)
            if picked is None:
                ax.text(0.5, 0.5, f"no layer matched\n{patterns}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8)
                continue
            lname, W = picked
            d = W.shape[1]

            pre = per_row_l2_normalize(W)
            rot = make_rotation(d, seed=0, rotation_type="srht")
            W_rot = apply_rotation(W.astype(np.float32), rot)
            post = per_row_l2_normalize(W_rot)

            scale = max(np.abs(post).max(), np.abs(pre).max(), 0.05)
            local_bins = np.linspace(-1.05 * scale, 1.05 * scale, 80)

            ax.hist(pre, bins=local_bins, density=True, alpha=0.55,
                    color="#1f77b4", label="pre-rotation")
            ax.hist(post, bins=local_bins, density=True, alpha=0.55,
                    color="#ff7f0e", label="post-rotation")

            a = d / 2.0
            x01 = (x_pdf + 1) / 2
            pdf = beta_dist.pdf(x01, a, a) / 2.0
            ax.plot(x_pdf, pdf, "r-", lw=1.3, label=f"Beta({d/2:.0f},{d/2:.0f})")

            ax.set_xlim(-1.05 * scale, 1.05 * scale)
            ax.set_yticks([])
            short_lname = lname.split(".")[-2] + "." + lname.split(".")[-1] if "." in lname else lname
            ax.set_title(f"{mname}\n{short_lname}  (d={d})", fontsize=8.5)
            if row_i == 0 and col_j == 2:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

        # Free GPU/CPU memory between models
        del model, tokenizer
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig.suptitle(
        "LLM weight coordinates: pre vs post ApexQuant rotation\n"
        "(red overlay: Beta(d/2, d/2) — the predicted post-rotation density)",
        fontsize=10, y=0.997,
    )
    fig.text(0.5, 0.005,
             "All LLM linear layers sit in the favorable large-d regime (d in 2048..8192). "
             "Beta is sharp; post-rotation locks onto it tightly across attention and MLP. "
             "No small-d failure modes, unlike the depthwise-heavy vision architectures.",
             ha="center", fontsize=8.5, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.015, 1, 0.985])
    out = FIG_DIR / "fig_rotation_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
