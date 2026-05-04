"""Orchestrator: sweep pretrained ImageNet models x bits x codebooks.

For each model:
  1. Load torchvision pretrained weights.
  2. Evaluate FP32 top-1 / top-5 on ImageNet val.
  3. For each (bits, codebook, variant in {baseline, apexquant}):
       - Quantize (baseline = direct; apexquant = rotate then quantize).
       - Evaluate top-1 / top-5 on ImageNet val.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from apexquant.ptq import quantize_model_baseline, quantize_model_apexquant, quantize_model_quarot

from imagenet_loader import get_imagenet_val_loader
from models import load_pretrained, model_size_at_bits

_THIS = Path(__file__).resolve().parent


RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.jsonl"


def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def eval_top1_top5(model: nn.Module, loader, device) -> tuple[float, float, int]:
    """Returns (top1, top5, n_images)."""
    model.eval()
    correct1, correct5, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        _, pred5 = logits.topk(5, dim=1)
        correct1 += (pred5[:, 0] == y).sum().item()
        correct5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    return correct1 / total, correct5 / total, total


def sweep_model(model_name: str, bits_list, codebooks, loader, device,
                rotation_seed: int = 0):
    print(f"\n=== {model_name} ===")
    model = load_pretrained(model_name).to(device)
    size_info = model_size_at_bits(model, None)
    print(f"  params: {size_info['n_params_total']:,}  "
          f"quantizable weights: {size_info['n_weights_quant']:,}  "
          f"FP32 size: {size_info['fp32_mb']:.1f} MB")

    # FP32 eval
    t0 = time.time()
    top1_fp32, top5_fp32, n = eval_top1_top5(model, loader, device)
    print(f"  FP32: top1={top1_fp32*100:.2f}%  top5={top5_fp32*100:.2f}%  "
          f"(n={n})  [{time.time()-t0:.0f}s]")
    append_result({
        "model": model_name,
        "variant": "fp32",
        "bits": 32,
        "codebook": "fp32",
        "top1": top1_fp32,
        "top5": top5_fp32,
        "n_images": n,
        "size_mb": size_info["fp32_mb"],
        "compression_ratio": 1.0,
        "fp32_top1": top1_fp32,
        "fp32_top5": top5_fp32,
        "elapsed_s": time.time() - t0,
    })

    def _log_eval(variant: str, bits: int, codebook: str, model_q, stats, s, t0):
        top1, top5, n = eval_top1_top5(model_q, loader, device)
        drop1 = top1_fp32 - top1
        drop5 = top5_fp32 - top5
        print(f"  {variant:8s} bits={bits}  cb={codebook:14s}  "
              f"top1={top1*100:6.2f}%  top5={top5*100:6.2f}%  "
              f"Δtop1={drop1*100:+.2f}pp  "
              f"size={s['quantized_mb']:.2f}MB ({s['ratio']:.1f}x)  "
              f"[{time.time()-t0:.0f}s]")
        append_result({
            "model": model_name,
            "variant": variant,
            "bits": bits,
            "codebook": codebook,
            "top1": top1,
            "top5": top5,
            "top1_drop": drop1,
            "top5_drop": drop5,
            "n_images": n,
            "size_mb": s["quantized_mb"],
            "compression_ratio": s["ratio"],
            "fp32_top1": top1_fp32,
            "fp32_top5": top5_fp32,
            "rotation_seed": rotation_seed,
            "layer_mse": [ls.mse for ls in stats],
            "layer_ks_D": [ls.ks_D for ls in stats],
            "elapsed_s": time.time() - t0,
        })

    # PTQ sweep
    for bits in bits_list:
        s = model_size_at_bits(model, bits)
        for codebook in codebooks:
            for variant in ["baseline", "apexquant"]:
                t0 = time.time()
                if variant == "baseline":
                    model_q, stats = quantize_model_baseline(model, bits, codebook)
                else:
                    model_q, stats = quantize_model_apexquant(
                        model, bits, codebook, rotation_seed=rotation_seed
                    )
                model_q = model_q.to(device)
                _log_eval(variant, bits, codebook, model_q, stats, s, t0)
                del model_q

        # QuaRot-style baseline (one eval per bit width; uniform absmax quant)
        t0 = time.time()
        model_q, stats = quantize_model_quarot(model, bits, rotation_seed=rotation_seed)
        model_q = model_q.to(device)
        _log_eval("quarot", bits, "symabs_uniform", model_q, stats, s, t0)
        del model_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["resnet18", "resnet50", "mobilenet_v2"])
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--codebooks", type=str, nargs="+",
                        default=["uniform", "beta"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=None,
                        help="use only first N val images (smoke tests)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}  Bits: {args.bits}  Codebooks: {args.codebooks}")

    loader = get_imagenet_val_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    for model_name in args.models:
        sweep_model(model_name, args.bits, args.codebooks, loader, device,
                    rotation_seed=args.rotation_seed)

    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
