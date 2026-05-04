"""ViT sweep runner: extends run.py to also quantize nn.MultiheadAttention.in_proj_weight.

Same baseline / ApexQuant / QuaRot variants and same codebooks as run.py, but applies
mha_quant.quantize_mha_in_place() after the main quantization step. This catches
the 33% of ViT weights that live in MultiheadAttention.in_proj_weight as raw
Parameters outside any nn.Linear module.

Size accounting is also corrected to include MHA weights in the quantized budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent

from apexquant.ptq import (  # noqa: E402
    quantize_model_baseline, quantize_model_apexquant, quantize_model_quarot,
)

from imagenet_loader import get_imagenet_val_loader
from models import load_pretrained
from mha_quant import quantize_mha_in_place, model_size_with_mha, has_mha


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


def sweep_vit(model_name: str, bits_list, codebooks, loader, device,
              rotation_seed: int = 0):
    print(f"\n=== {model_name} (with MHA quantization) ===")
    model = load_pretrained(model_name).to(device)
    info = model_size_with_mha(model, 32)
    print(f"  params: {info['n_params_total']:,}")
    print(f"  weights via Conv2d/Linear: {info['n_weights_main']:,}")
    print(f"  weights via MHA in_proj:   {info['n_weights_mha']:,}")
    print(f"  total quantizable:          {info['n_weights_quant']:,}")
    print(f"  FP32 size: {info['fp32_mb']:.1f} MB")

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
        "size_mb": info["fp32_mb"],
        "compression_ratio": 1.0,
        "fp32_top1": top1_fp32,
        "fp32_top5": top5_fp32,
        "elapsed_s": time.time() - t0,
        "mha_quantized": True,
    })

    for bits in bits_list:
        s = model_size_with_mha(model, bits)
        # main + mha quantization for each (variant, codebook)
        for codebook in codebooks:
            for variant_label in ["baseline", "apexquant"]:
                t0 = time.time()
                if variant_label == "baseline":
                    model_q, stats = quantize_model_baseline(model, bits, codebook)
                else:
                    model_q, stats = quantize_model_apexquant(
                        model, bits, codebook, rotation_seed=rotation_seed)
                quantize_mha_in_place(model_q, variant_label, bits,
                                       codebook=codebook, rotation_seed=rotation_seed)
                model_q = model_q.to(device)
                top1, top5, n = eval_top1_top5(model_q, loader, device)
                drop1 = top1_fp32 - top1
                drop5 = top5_fp32 - top5
                print(f"  {variant_label:8s} bits={bits} cb={codebook:8s}  "
                      f"top1={top1*100:6.2f}%  top5={top5*100:6.2f}%  "
                      f"Δtop1={drop1*100:+.2f}pp  "
                      f"size={s['quantized_mb']:.2f}MB ({s['ratio']:.1f}x)  "
                      f"[{time.time()-t0:.0f}s]")
                append_result({
                    "model": model_name,
                    "variant": variant_label,
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
                    "mha_quantized": True,
                })
                del model_q

        # QuaRot variant
        t0 = time.time()
        model_q, stats = quantize_model_quarot(model, bits,
                                                rotation_seed=rotation_seed)
        quantize_mha_in_place(model_q, "quarot", bits,
                               rotation_seed=rotation_seed)
        model_q = model_q.to(device)
        top1, top5, n = eval_top1_top5(model_q, loader, device)
        drop1 = top1_fp32 - top1
        drop5 = top5_fp32 - top5
        print(f"  {'quarot':8s} bits={bits} cb={'symabs':8s}  "
              f"top1={top1*100:6.2f}%  top5={top5*100:6.2f}%  "
              f"Δtop1={drop1*100:+.2f}pp  "
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
            "rotation_seed": rotation_seed,
            "layer_mse": [ls.mse for ls in stats],
            "layer_ks_D": [ls.ks_D for ls in stats],
            "elapsed_s": time.time() - t0,
            "mha_quantized": True,
        })
        del model_q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vit_b_16"])
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--codebooks", nargs="+", default=["uniform", "beta"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rotation-seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}  Bits: {args.bits}  Codebooks: {args.codebooks}")

    loader = get_imagenet_val_loader(batch_size=args.batch_size,
                                     num_workers=args.num_workers)

    for name in args.models:
        sweep_vit(name, args.bits, args.codebooks, loader, device,
                  rotation_seed=args.rotation_seed)

    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
