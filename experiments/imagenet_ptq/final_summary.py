"""Aggregate final results across all 6 models and render tables + figures."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path(__file__).resolve().parent / "results" / "results.jsonl"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with open(RESULTS_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def dedupe_latest(rows):
    """For each (model, variant, bits, codebook, seed) keep the latest row."""
    seen = {}
    for r in rows:
        key = (
            r["model"], r["variant"], r["bits"], r["codebook"],
            r.get("rotation_seed", 0),
        )
        seen[key] = r
    return list(seen.values())


MODEL_ORDER = [
    "resnet18", "resnet50", "mobilenet_v2",
    "vit_b_16", "convnext_tiny", "efficientnet_b0",
]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNet-V2",
    "vit_b_16": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b0": "EfficientNet-B0",
}


def _best(rows, model, bits, variants_codebooks):
    """Return top-1 of first matching (variant, codebook) from the list, or None."""
    for v, cb in variants_codebooks:
        for r in rows:
            if (r["model"] == model and r["variant"] == v
                    and r["bits"] == bits and r["codebook"] == cb
                    and r.get("rotation_seed", 0) == 0):
                return r["top1"]
    return None


def print_master_tables(rows):
    """Six markdown tables, one per model."""
    fp32 = {}
    for r in rows:
        if r["variant"] == "fp32":
            fp32[r["model"]] = r

    for model in MODEL_ORDER:
        if model not in fp32:
            continue
        f = fp32[model]
        print()
        print("=" * 90)
        print(f"## {MODEL_LABELS[model]}  —  "
              f"FP32: {f['top1']*100:.2f}% top-1 / {f['top5']*100:.2f}% top-5 / "
              f"{f['size_mb']:.1f} MB")
        print("=" * 90)
        header = (f"{'bits':>4} | {'cb':>7} | {'size MB':>8} | {'ratio':>6} | "
                 f"{'baseline':>9} | {'QuaRot':>9} | {'H2 (ours)':>10}")
        print(header)
        print("-" * len(header))
        for bits in [2, 4, 6, 8]:
            for cb in ["uniform", "beta"]:
                base = _best(rows, model, bits, [("baseline", cb)])
                h2 = _best(rows, model, bits, [("h2", cb)])
                size, ratio = None, None
                for r in rows:
                    if (r["model"] == model and r["bits"] == bits
                            and r.get("rotation_seed", 0) == 0
                            and r["variant"] in ("baseline", "h2")
                            and r["codebook"] == cb):
                        size = r["size_mb"]
                        ratio = r["compression_ratio"]
                        break
                if cb == "beta":
                    # QuaRot is codebook-independent; show under beta row for comparison
                    q = _best(rows, model, bits, [("quarot", "symabs_uniform")])
                    row = (f"{bits:>4} | {cb:>7} | "
                           f"{size:>8.2f} | {ratio:>5.1f}x | "
                           f"{(base*100):>8.2f}% | "
                           f"{(q*100 if q is not None else 0):>8.2f}% | "
                           f"{(h2*100 if h2 is not None else 0):>9.2f}%")
                else:
                    row = (f"{bits:>4} | {cb:>7} | "
                           f"{size:>8.2f} | {ratio:>5.1f}x | "
                           f"{(base*100):>8.2f}% | "
                           f"{'—':>9} | "
                           f"{(h2*100 if h2 is not None else 0):>9.2f}%")
                print(row)


def fig_top1_vs_bits_all(rows):
    """6 subplots, one per model. Lines: baseline_beta, quarot, h2_beta."""
    fp32 = {r["model"]: r["top1"] for r in rows if r["variant"] == "fp32"}
    n = len(MODEL_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    styles = {
        "baseline_beta": {"color": "#888", "marker": "o", "linestyle": "--",
                          "label": "Baseline + Beta"},
        "quarot": {"color": "#1f77b4", "marker": "s", "linestyle": "-",
                   "label": "QuaRot"},
        "h2_beta": {"color": "#d62728", "marker": "^", "linestyle": "-",
                    "label": "H2 + Beta (ours)", "linewidth": 2},
    }

    for ax, model in zip(axes, MODEL_ORDER):
        if model not in fp32:
            continue
        pts = {"baseline_beta": [], "quarot": [], "h2_beta": []}
        for bits in [2, 4, 6, 8]:
            b = _best(rows, model, bits, [("baseline", "beta")])
            q = _best(rows, model, bits, [("quarot", "symabs_uniform")])
            h = _best(rows, model, bits, [("h2", "beta")])
            if b is not None: pts["baseline_beta"].append((bits, b * 100))
            if q is not None: pts["quarot"].append((bits, q * 100))
            if h is not None: pts["h2_beta"].append((bits, h * 100))
        for key, style in styles.items():
            ps = pts[key]
            if not ps: continue
            bits, ys = zip(*ps)
            ax.plot(bits, ys, **style, markersize=6)
        ax.axhline(fp32[model] * 100, color="k", linestyle=":",
                   linewidth=1, alpha=0.5, label=f"FP32 ({fp32[model]*100:.1f}%)")
        ax.set_xlabel("bits per weight")
        ax.set_ylabel("top-1 accuracy (%)")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Post-training weight quantization on ImageNet val (50k images)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_imagenet_all_models.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


def fig_rotation_gain_all(rows):
    """Rotation gain (H2 - Baseline, Beta codebook) vs bits, per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    for color, model in zip(colors, MODEL_ORDER):
        pts = []
        for bits in [2, 4, 6, 8]:
            b = _best(rows, model, bits, [("baseline", "beta")])
            h = _best(rows, model, bits, [("h2", "beta")])
            if b is None or h is None: continue
            pts.append((bits, (h - b) * 100))
        if not pts: continue
        bits, gains = zip(*pts)
        ax.plot(bits, gains, marker="o", color=color,
                label=MODEL_LABELS[model], linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("rotation gain over baseline (Beta codebook, pp)")
    ax.set_title("Rotation gain: H2+Beta − Baseline+Beta")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_imagenet_rotation_gain_all.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig_ours_vs_quarot(rows):
    """Our advantage (H2+Beta − QuaRot) vs bits, per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    for color, model in zip(colors, MODEL_ORDER):
        pts = []
        for bits in [2, 4, 6, 8]:
            q = _best(rows, model, bits, [("quarot", "symabs_uniform")])
            h = _best(rows, model, bits, [("h2", "beta")])
            if q is None or h is None: continue
            pts.append((bits, (h - q) * 100))
        if not pts: continue
        bits, gains = zip(*pts)
        ax.plot(bits, gains, marker="o", color=color,
                label=MODEL_LABELS[model], linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("ours over QuaRot (pp)")
    ax.set_title("Method-to-method gap: (H2 + Beta) − QuaRot")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_imagenet_ours_vs_quarot.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    rows_raw = load_rows()
    rows = dedupe_latest(rows_raw)
    print(f"Loaded {len(rows_raw)} rows → deduped to {len(rows)}")
    print_master_tables(rows)
    fig_top1_vs_bits_all(rows)
    fig_rotation_gain_all(rows)
    fig_ours_vs_quarot(rows)


if __name__ == "__main__":
    main()
