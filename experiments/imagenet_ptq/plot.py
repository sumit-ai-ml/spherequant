"""Figures + summary table for the ImageNet PTQ sweep."""

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
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_top1_vs_bits(rows):
    """One subplot per model. X = bits. Y = top-1. Lines: baseline uniform,
    baseline beta, SphereQuant uniform, SphereQuant beta. Dotted = FP32 reference."""
    rows_ptq = [r for r in rows if r["variant"] != "fp32"]
    fp32_by_model = {r["model"]: r["top1"] for r in rows if r["variant"] == "fp32"}
    models = sorted(fp32_by_model.keys())

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=False)
    if n_models == 1:
        axes = [axes]

    styles = {
        ("baseline", "uniform"): {"color": "#888888", "marker": "o", "linestyle": "-"},
        ("baseline", "beta"): {"color": "#888888", "marker": "o", "linestyle": "--"},
        ("spherequant", "uniform"): {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
        ("spherequant", "beta"): {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
    }

    for ax, model in zip(axes, models):
        for (variant, codebook), style in styles.items():
            pts = [(r["bits"], r["top1"] * 100) for r in rows_ptq
                   if r["model"] == model and r["variant"] == variant
                   and r["codebook"] == codebook]
            if not pts:
                continue
            pts.sort()
            bits, top1 = zip(*pts)
            ax.plot(bits, top1, label=f"{variant} / {codebook}",
                    linewidth=1.5, markersize=5, **style)
        ax.axhline(fp32_by_model[model] * 100, color="k", linestyle=":",
                   linewidth=1, alpha=0.6, label="FP32 reference")
        ax.set_xlabel("bits per coordinate")
        ax.set_ylabel("top-1 accuracy (%)")
        ax.set_title(f"{model}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("ImageNet post-training quantization: top-1 vs bits")
    fig.tight_layout()
    out = FIG_DIR / "fig_imagenet_top1_vs_bits.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def fig_rotation_gain(rows):
    """How much does SphereQuant improve over baseline? per (model, codebook, bits)."""
    rows_ptq = [r for r in rows if r["variant"] != "fp32"]
    # (model, bits, codebook, variant) -> top1
    tbl = {(r["model"], r["bits"], r["codebook"], r["variant"]): r["top1"]
           for r in rows_ptq}
    models = sorted({r["model"] for r in rows_ptq})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, codebook in zip(axes, ["uniform", "beta"]):
        x_positions = []
        for i, model in enumerate(models):
            bits_list = sorted({r["bits"] for r in rows_ptq
                                if r["model"] == model
                                and r["codebook"] == codebook})
            gains = []
            bits_kept = []
            for b in bits_list:
                base = tbl.get((model, b, codebook, "baseline"))
                sq = tbl.get((model, b, codebook, "spherequant"))
                if base is None or sq is None:
                    continue
                gains.append(100 * (sq - base))
                bits_kept.append(b)
            if not gains:
                continue
            ax.plot(bits_kept, gains, marker="o", label=model)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("bits per coordinate")
        ax.set_title(f"codebook = {codebook}")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("top-1 gain from rotation (pp)")
    fig.suptitle("Accuracy gain from TurboQuant-style rotation (SphereQuant - baseline)")
    fig.tight_layout()
    out = FIG_DIR / "fig_imagenet_rotation_gain.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def print_summary(rows):
    fp32_by_model = {r["model"]: (r["top1"], r["top5"], r["size_mb"])
                     for r in rows if r["variant"] == "fp32"}
    rows_ptq = [r for r in rows if r["variant"] != "fp32"]
    models = sorted({r["model"] for r in rows_ptq})

    print()
    print("=" * 120)
    print(f"{'model':15s} {'variant':8s} {'cb':8s} {'bits':>4s} "
          f"{'size MB':>8s} {'comp':>6s} "
          f"{'top1 %':>7s} {'top5 %':>7s} "
          f"{'Δtop1':>7s} {'Δtop5':>7s}")
    print("=" * 120)
    for model in models:
        t1f, t5f, sf = fp32_by_model[model]
        print(f"{model:15s} {'fp32':8s} {'-':8s} {'32':>4s} "
              f"{sf:>8.1f} {'1.0x':>6s} "
              f"{t1f*100:>7.2f} {t5f*100:>7.2f} "
              f"{'-':>7s} {'-':>7s}")
        for variant in ["baseline", "spherequant"]:
            for codebook in ["uniform", "beta"]:
                for bits in [8, 6, 4, 2]:
                    r = next((r for r in rows_ptq
                              if r["model"] == model
                              and r["variant"] == variant
                              and r["codebook"] == codebook
                              and r["bits"] == bits), None)
                    if r is None:
                        continue
                    print(f"{'':15s} {variant:8s} {codebook:8s} {bits:>4d} "
                          f"{r['size_mb']:>8.2f} "
                          f"{r['compression_ratio']:>5.1f}x "
                          f"{r['top1']*100:>7.2f} {r['top5']*100:>7.2f} "
                          f"{-r['top1_drop']*100:>+7.2f} "
                          f"{-r['top5_drop']*100:>+7.2f}")
        print("-" * 120)


def main():
    if not RESULTS_FILE.exists():
        print(f"No results at {RESULTS_FILE}. Run run.py first.")
        return
    rows = load_rows()
    print(f"Loaded {len(rows)} rows.")
    fig_top1_vs_bits(rows)
    fig_rotation_gain(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
