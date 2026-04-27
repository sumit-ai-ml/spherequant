"""Aggregate LLM results + render figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path(__file__).resolve().parent / "results" / "results.jsonl"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with open(RESULTS_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


METHOD_LABELS = {
    "rtn_absmax": "RTN (per-channel absmax, no rotation)",
    "quarot": "QuaRot (rotation + absmax uniform)",
    "spherequant": "SphereQuant + Beta (ours)",
}
METHOD_STYLES = {
    "rtn_absmax": {"color": "#888", "marker": "o", "linestyle": "-"},
    "quarot": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "spherequant": {"color": "#d62728", "marker": "^", "linestyle": "-", "linewidth": 2},
}


def fig_ppl_vs_bits(rows):
    fp16 = next((r for r in rows if r["variant"] == "fp16"), None)
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: linear ppl for 4+ bits (removes the 2-bit outliers so readable)
    ax = axes[0]
    for method, style in METHOD_STYLES.items():
        pts = [(r["bits"], r["perplexity"]) for r in rows
               if r["variant"] == method and r["bits"] >= 4]
        if not pts: continue
        pts.sort()
        bits, ys = zip(*pts)
        ax.plot(bits, ys, label=METHOD_LABELS[method], markersize=6, **style)
    if fp16:
        ax.axhline(fp16["perplexity"], color="k", linestyle=":", linewidth=1,
                   label=f"FP16 ({fp16['perplexity']:.2f})")
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_title("4/6/8-bit PTQ (linear scale)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Right: log ppl for all bits (shows 2-bit collapse)
    ax = axes[1]
    for method, style in METHOD_STYLES.items():
        pts = [(r["bits"], r["perplexity"]) for r in rows
               if r["variant"] == method]
        if not pts: continue
        pts.sort()
        bits, ys = zip(*pts)
        ax.semilogy(bits, ys, label=METHOD_LABELS[method], markersize=6, **style)
    if fp16:
        ax.axhline(fp16["perplexity"], color="k", linestyle=":", linewidth=1,
                   label=f"FP16 ({fp16['perplexity']:.2f})")
    ax.set_xlabel("bits per weight")
    ax.set_ylabel("WikiText-2 perplexity (log scale)")
    ax.set_title("Full sweep 2-8 bit (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    title = models[0] if models else "LLM"
    fig.suptitle(f"LLM PTQ on {title} — WikiText-2", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_llm_ppl_vs_bits.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def print_table(rows):
    fp16 = next((r for r in rows if r["variant"] == "fp16"), None)
    fp16_ppl = fp16["perplexity"] if fp16 else None
    print(f"\n{'='*90}")
    print(f"## TinyLlama 1.1B  —  FP16 perplexity: {fp16_ppl:.3f} / {fp16['size_mb']:.1f} MB")
    print("="*90)
    print(f"{'bits':>4} | {'method':>16} | {'size MB':>8} | {'ratio':>6} | "
          f"{'perplexity':>12} | {'% change':>9}")
    print("-"*90)
    for bits in [8, 6, 4, 2]:
        for method in ["rtn_absmax", "quarot", "spherequant"]:
            r = next((r for r in rows
                      if r["variant"] == method and r["bits"] == bits), None)
            if r is None: continue
            pct = (r["perplexity"] - fp16_ppl) / fp16_ppl * 100
            pct_str = f"{pct:+.1f}%" if abs(pct) < 1000 else f"{pct:+.0e}"
            print(f"{bits:>4} | {method:>16} | "
                  f"{r['size_mb']:>8.1f} | {r['compression_ratio']:>5.2f}x | "
                  f"{r['perplexity']:>12.3f} | {pct_str:>9}")
        print()


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows.")
    print_table(rows)
    fig_ppl_vs_bits(rows)


if __name__ == "__main__":
    main()
