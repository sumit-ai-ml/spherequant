"""Append-only QuaRot baseline runner.

Runs quantize_model_quarot for each (model, bits) combination and appends
results to results/results.jsonl with variant="quarot". Does NOT touch any
rows already written. Use this after run.py has finished the main sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent

from apexquant.ptq import quantize_model_quarot  # noqa: E402

from imagenet_loader import get_imagenet_val_loader
from models import load_pretrained, model_size_at_bits


RESULTS_DIR = _THIS / "results"
RESULTS_FILE = RESULTS_DIR / "results.jsonl"


def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def eval_top1_top5(model, loader, device):
    model.eval()
    c1, c5, t = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        _, p5 = logits.topk(5, dim=1)
        c1 += (p5[:, 0] == y).sum().item()
        c5 += (p5 == y.unsqueeze(1)).any(dim=1).sum().item()
        t += y.numel()
    return c1 / t, c5 / t, t


def existing_fp32(model_name: str) -> tuple[float, float] | None:
    if not RESULTS_FILE.exists():
        return None
    for line in RESULTS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("model") == model_name and r.get("variant") == "fp32":
            return r["top1"], r["top5"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["resnet18", "resnet50", "mobilenet_v2"])
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rotation-seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}  Bits: {args.bits}")

    loader = get_imagenet_val_loader(batch_size=args.batch_size,
                                     num_workers=args.num_workers)

    for model_name in args.models:
        print(f"\n=== {model_name} (QuaRot) ===")
        fp32 = existing_fp32(model_name)
        model = load_pretrained(model_name).to(device)
        if fp32 is None:
            top1_fp32, top5_fp32, _ = eval_top1_top5(model, loader, device)
            print(f"  FP32 (recomputed): top1={top1_fp32*100:.2f}%  "
                  f"top5={top5_fp32*100:.2f}%")
        else:
            top1_fp32, top5_fp32 = fp32
            print(f"  FP32 (from existing results): top1={top1_fp32*100:.2f}%  "
                  f"top5={top5_fp32*100:.2f}%")
        for bits in args.bits:
            s = model_size_at_bits(model, bits)
            t0 = time.time()
            model_q, stats = quantize_model_quarot(model, bits,
                                                   rotation_seed=args.rotation_seed)
            model_q = model_q.to(device)
            top1, top5, n = eval_top1_top5(model_q, loader, device)
            drop1 = top1_fp32 - top1
            drop5 = top5_fp32 - top5
            print(f"  quarot bits={bits}  top1={top1*100:6.2f}%  "
                  f"top5={top5*100:6.2f}%  Δtop1={drop1*100:+.2f}pp  "
                  f"size={s['quantized_mb']:.2f}MB ({s['ratio']:.1f}x)  "
                  f"[{time.time()-t0:.0f}s]")
            append_result({
                "model": model_name,
                "variant": "quarot",
                "bits": bits,
                "codebook": "symabs_uniform",
                "top1": top1,
                "top5": top5,
                "top1_drop": drop1,
                "top5_drop": drop5,
                "n_images": n,
                "size_mb": s["quantized_mb"],
                "compression_ratio": s["ratio"],
                "fp32_top1": top1_fp32,
                "fp32_top5": top5_fp32,
                "rotation_seed": args.rotation_seed,
                "layer_mse": [ls.mse for ls in stats],
                "layer_ks_D": [ls.ks_D for ls in stats],
                "elapsed_s": time.time() - t0,
            })
            del model_q

    print(f"\nDone. Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
