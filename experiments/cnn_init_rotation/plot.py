"""Plot figures from results/results.jsonl.

Three figures:

  fig_acc_vs_bits.png
    Mean accuracy drop vs bits for each (variant, codebook), seeds averaged.
    This is the headline comparison: does rotation (ApexQuant) or rotated-basis
    training (H3) reduce the accuracy drop from quantization?

  fig_weight_histograms.png
    Density of weight coordinates for each layer under three views:
      - baseline: flattened W_flat (per-row L2-normalized) coords
      - ApexQuant:       baseline weight after post-hoc rotation, normalized rows
      - H3:       trained U (already rotated) normalized rows
    Overlay Beta(d/2, d/2) PDF for reference.

  fig_ks_vs_layer.png
    KS statistic against Beta(d/2, d/2) per layer for each variant.
    Lower = better fit to TurboQuant's distributional assumption.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

import torch

from model import CNN3
from rotated_conv import RotatedCNN3
from apexquant.rotation_utils import make_rotation, apply_rotation, SRHTRotation


RESULTS_FILE = Path(__file__).resolve().parent / "results" / "results.jsonl"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path = RESULTS_FILE):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- Generic drop-vs-bits helper ----------

def _plot_drop_vs_bits(rows, metric_key: str, ylabel: str, title: str,
                       out_name: str, scale: float = 1.0, fmt: str = "{:+.2f}"):
    """metric_key is e.g. 'acc_drop' or 'auroc_drop'. scale multiplies the
    values (100 to convert to pp for accuracy; 1 for AUROC)."""
    groups = defaultdict(list)  # (variant, codebook, bits) -> [drop]
    for r in rows:
        if metric_key not in r:
            continue
        groups[(r["variant"], r["codebook"], r["bits"])].append(r[metric_key])

    if not groups:
        print(f"No rows with {metric_key}; skipping {out_name}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, codebook in zip(axes, ["uniform", "beta"]):
        for variant, marker, color in [
            ("baseline", "o", "#888888"),
            ("apexquant", "s", "#1f77b4"),
            ("h3", "^", "#d62728"),
        ]:
            bits_list, mean_drop, std_drop = [], [], []
            for (v, c, b), drops in sorted(groups.items()):
                if v == variant and c == codebook:
                    bits_list.append(b)
                    mean_drop.append(scale * np.mean(drops))
                    std_drop.append(scale * np.std(drops))
            if not bits_list:
                continue
            bits_list = np.array(bits_list)
            order = np.argsort(bits_list)
            bits_list = bits_list[order]
            mean_drop = np.array(mean_drop)[order]
            std_drop = np.array(std_drop)[order]
            ax.errorbar(bits_list, mean_drop, yerr=std_drop, marker=marker,
                        color=color, label=variant, capsize=3, linewidth=1.5)
        ax.set_xlabel("bits per coordinate")
        ax.set_title(f"codebook = {codebook}")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig_acc_vs_bits(rows):
    _plot_drop_vs_bits(
        rows, metric_key="acc_drop",
        ylabel="test accuracy drop (pp)",
        title="Post-training quantization: accuracy drop vs bits",
        out_name="fig_acc_vs_bits.png", scale=100.0,
    )


def fig_auroc_vs_bits(rows):
    _plot_drop_vs_bits(
        rows, metric_key="auroc_drop",
        ylabel="macro AUROC drop",
        title="Post-training quantization: macro-AUROC drop vs bits",
        out_name="fig_auroc_vs_bits.png", scale=1.0,
    )


def fig_auroc_absolute(rows):
    """Absolute AUROC (not drop) vs bits, so you can see where the Beta codebook
    pulls quantized models back up to the FP32 line."""
    groups = defaultdict(list)
    fp32_by_variant = defaultdict(list)
    for r in rows:
        if "quant_auroc" not in r:
            continue
        groups[(r["variant"], r["codebook"], r["bits"])].append(r["quant_auroc"])
        fp32_by_variant[r["variant"]].append(r["fp32_auroc"])
    if not groups:
        print("No AUROC rows; skipping absolute plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, codebook in zip(axes, ["uniform", "beta"]):
        for variant, marker, color in [
            ("baseline", "o", "#888888"),
            ("apexquant", "s", "#1f77b4"),
            ("h3", "^", "#d62728"),
        ]:
            bits_list, mean, std = [], [], []
            for (v, c, b), vals in sorted(groups.items()):
                if v == variant and c == codebook:
                    bits_list.append(b)
                    mean.append(np.mean(vals))
                    std.append(np.std(vals))
            if not bits_list:
                continue
            order = np.argsort(bits_list)
            bits_list = np.array(bits_list)[order]
            mean = np.array(mean)[order]
            std = np.array(std)[order]
            ax.errorbar(bits_list, mean, yerr=std, marker=marker, color=color,
                        label=variant, capsize=3, linewidth=1.5)
            # FP32 reference line for this variant
            if fp32_by_variant[variant]:
                fp32_mean = np.mean(fp32_by_variant[variant])
                ax.axhline(fp32_mean, color=color, linestyle=":", linewidth=1,
                           alpha=0.5)
        ax.set_xlabel("bits per coordinate")
        ax.set_title(f"codebook = {codebook}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro AUROC")
    axes[0].legend(loc="lower right")
    fig.suptitle("Absolute macro-AUROC vs bits (dotted = FP32 reference)")
    fig.tight_layout()
    out = FIG_DIR / "fig_auroc_absolute.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# ---------- Figure 2: weight histograms ----------

def _unit_row_coords(W_flat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W_flat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (W_flat / norms).ravel()


def fig_weight_histograms(seed: int = 0):
    """Pick a single seed and show coordinate distributions per layer."""
    # We need models to examine. Quickest: build and load from a fresh
    # init then train a few batches? That's wasteful. Instead, we reuse
    # the per-row normalized distribution of Kaiming-initialized weights,
    # which is what matters for the KS/Beta comparison claim.
    torch.manual_seed(seed)
    baseline = CNN3()
    rotated = RotatedCNN3(seed=seed)

    layers_base = [(n, m.weight.detach().cpu().numpy().reshape(m.weight.shape[0], -1))
                   for n, m in baseline.named_modules()
                   if hasattr(m, "weight") and m.weight is not None
                   and m.weight.ndim >= 2]
    layers_h3 = [(n, m.U.detach().cpu().numpy())
                 for n, m in rotated.rotated_layers()]

    n = len(layers_base)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)
    for i, ((name_b, Wb), (name_h, Uh)) in enumerate(zip(layers_base, layers_h3)):
        ax = axes[i, 0]
        d = Wb.shape[1]

        # baseline normalized coords
        coords_b = _unit_row_coords(Wb)

        # ApexQuant: rotate baseline then normalize
        rot = make_rotation(d, seed * 1000 + i + 1, "srht")
        Wb_rot = apply_rotation(Wb.astype(np.float32), rot)
        coords_h2 = _unit_row_coords(Wb_rot)

        # H3: normalized U rows
        coords_h3 = _unit_row_coords(Uh)

        bins = np.linspace(-1, 1, 80)
        ax.hist(coords_b, bins=bins, density=True, alpha=0.35, color="#888888",
                label="baseline W (unit-row)")
        ax.hist(coords_h2, bins=bins, density=True, alpha=0.35, color="#1f77b4",
                label="ApexQuant rot(W) (unit-row)")
        ax.hist(coords_h3, bins=bins, density=True, alpha=0.35, color="#d62728",
                label="H3 U (unit-row)")

        # Overlay Beta(d/2, d/2) on [-1, 1]
        xs = np.linspace(-0.999, 0.999, 400)
        xs01 = (xs + 1) / 2
        pdf = beta_dist.pdf(xs01, d / 2, d / 2) / 2  # /2 for Jacobian to [-1,1]
        ax.plot(xs, pdf, "k--", linewidth=1.2, label=f"Beta({d}/2,{d}/2)")
        ax.set_title(f"{name_b}  (d={d})")
        ax.set_xlim(-1, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Coordinate distributions after per-row L2 normalization (at init)")
    fig.tight_layout()
    out = FIG_DIR / "fig_weight_histograms.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# ---------- Figure 3: KS vs layer ----------

def fig_ks_vs_layer(rows):
    """Per-layer KS statistic against Beta(d/2, d/2).

    Baseline/ApexQuant use CNN3's nn.Sequential names (features.0/3/6, classifier);
    H3 uses RotatedCNN3's names (conv1/2/3, fc). We canonicalize by depth
    index 1..4 so the three variants are comparable in the same chart.
    """
    # Collect per-(variant, codebook=beta, bits=8) — KS doesn't depend on
    # bits/codebook, we just need one row per (variant, seed).
    picked = defaultdict(list)
    for r in rows:
        if r["bits"] != 8 or r["codebook"] != "beta":
            continue
        per_layer = [(s["name"], s["ks_D"], s["d"]) for s in r["layer_stats"]]
        picked[r["variant"]].append(per_layer)

    if not picked:
        print("No rows for bits=8/beta found; skipping KS plot.")
        return

    # Canonical depth labels (conv1, conv2, conv3, fc) with their d values.
    # Both CNN3 and RotatedCNN3 have 4 quantizable layers in the same order.
    variants = ["baseline", "apexquant", "h3"]
    n_layers = len(next(iter(picked.values()))[0])
    depth_labels = [f"L{i+1}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_layers)
    width = 0.27
    d_values = None
    for i, v in enumerate(variants):
        if v not in picked:
            continue
        means, stds = [], []
        for depth in range(n_layers):
            ks_at_depth = [row[depth][1] for row in picked[v]]
            means.append(np.mean(ks_at_depth))
            stds.append(np.std(ks_at_depth))
        if d_values is None:
            d_values = [picked[v][0][depth][2] for depth in range(n_layers)]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=v, capsize=3)

    # Annotate x-axis with layer depth + d
    xlabels = [f"{lab}\n(d={d})" for lab, d in zip(depth_labels, d_values)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("KS statistic vs Beta(d/2, d/2)")
    ax.set_title("Per-layer distributional fit after per-row L2 normalize")
    ax.axhline(0.05, linestyle="--", color="k", linewidth=0.5, alpha=0.5,
               label="poor-fit threshold (D=0.05)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_ks_vs_layer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    rows = load_results()
    if not rows:
        print(f"No results in {RESULTS_FILE}. Run run.py first.")
        return
    print(f"Loaded {len(rows)} result rows.")
    fig_acc_vs_bits(rows)
    fig_auroc_vs_bits(rows)
    fig_auroc_absolute(rows)
    fig_weight_histograms(seed=0)
    fig_ks_vs_layer(rows)


if __name__ == "__main__":
    main()
