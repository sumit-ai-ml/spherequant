"""Shared result container, summary printer, and JSONL writer for bench."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchResult:
    """One row of bench output. Vision uses top1/top5; LLM uses perplexity."""

    model: str
    task: str            # "image_classification" or "causal_lm"
    dataset: str
    variant: str         # "reference", "apexquant", "quarot", "rtn_absmax", "baseline"
    bits: int            # 32 / 16 for reference, else target bits
    codebook: str        # "fp32"/"fp16" for reference, else "beta"/"uniform"/"symabs_uniform"
    n_samples: int

    # Metrics — only one set is populated depending on task.
    top1: Optional[float] = None
    top5: Optional[float] = None
    perplexity: Optional[float] = None

    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)


def write_jsonl(results: list[BenchResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def format_summary(results: list[BenchResult]) -> str:
    """Print a compact summary table. Reference row first, quantized rows
    grouped by (variant, bits) ascending."""
    if not results:
        return "(no results)"

    task = results[0].task
    lines = []
    lines.append("")
    lines.append(f"=== Summary: {results[0].model}  on  {results[0].dataset} ===")

    if task == "image_classification":
        lines.append(f"{'variant':12s}  {'bits':>4}  {'codebook':16s}  "
                     f"{'top1':>8}  {'top5':>8}  {'Δtop1':>8}  {'time(s)':>8}")
        lines.append("-" * 72)
        ref = next((r for r in results if r.variant == "reference"), None)
        ref_top1 = ref.top1 if ref else None
        for r in results:
            drop = "" if (ref_top1 is None or r.top1 is None) \
                   else f"{(r.top1 - ref_top1) * 100:+7.2f}"
            lines.append(
                f"{r.variant:12s}  {r.bits:>4}  {r.codebook:16s}  "
                f"{(r.top1 or 0) * 100:7.2f}%  {(r.top5 or 0) * 100:7.2f}%  "
                f"{drop:>8}  {r.elapsed_s:>8.1f}"
            )
    else:  # causal_lm
        lines.append(f"{'variant':12s}  {'bits':>4}  {'codebook':16s}  "
                     f"{'perplexity':>12}  {'time(s)':>8}")
        lines.append("-" * 60)
        for r in results:
            lines.append(
                f"{r.variant:12s}  {r.bits:>4}  {r.codebook:16s}  "
                f"{(r.perplexity or 0):>12.3f}  {r.elapsed_s:>8.1f}"
            )

    lines.append("")
    return "\n".join(lines)
