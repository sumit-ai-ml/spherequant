"""Multi-seed error bars for specific (model, bits, codebook, variant) settings.

Used to add seed=1,2 runs on top of the seed=0 results already in results.jsonl.
Reports mean +- std explicitly to the log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_CNN_EXP = _THIS.parent / "cnn_init_rotation"
sys.path.insert(0, str(_CNN_EXP))

from ptq import quantize_model_h2  # noqa: E402

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


def existing_fp32(model_name: str):
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
    p.add_argument("--models", nargs="+", default=["resnet50"])
    p.add_argument("--bits", type=int, nargs="+", default=[4])
    p.add_argument("--codebooks", nargs="+", default=["beta"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2],
                   help="additional seeds to run (0 is already in results)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = get_imagenet_val_loader(batch_size=args.batch_size,
                                     num_workers=args.num_workers)

    for model_name in args.models:
        print(f"\n=== {model_name} multi-seed ===")
        model = load_pretrained(model_name).to(device)
        fp32 = existing_fp32(model_name)
        top1_fp32, top5_fp32 = fp32
        print(f"  FP32 ref: top1={top1_fp32*100:.2f}%  top5={top5_fp32*100:.2f}%")
        for bits in args.bits:
            for cb in args.codebooks:
                s = model_size_at_bits(model, bits)
                seed_top1s = []
                for seed in args.seeds:
                    t0 = time.time()
                    model_q, stats = quantize_model_h2(
                        model, bits, cb, rotation_seed=seed)
                    model_q = model_q.to(device)
                    top1, top5, n = eval_top1_top5(model_q, loader, device)
                    drop1 = top1_fp32 - top1
                    drop5 = top5_fp32 - top5
                    print(f"  h2 bits={bits} cb={cb} seed={seed}  "
                          f"top1={top1*100:6.2f}%  top5={top5*100:6.2f}%  "
                          f"Δtop1={drop1*100:+.2f}pp  "
                          f"[{time.time()-t0:.0f}s]")
                    append_result({
                        "model": model_name,
                        "variant": "h2",
                        "bits": bits,
                        "codebook": cb,
                        "top1": top1,
                        "top5": top5,
                        "top1_drop": drop1,
                        "top5_drop": drop5,
                        "n_images": n,
                        "size_mb": s["quantized_mb"],
                        "compression_ratio": s["ratio"],
                        "fp32_top1": top1_fp32,
                        "fp32_top5": top5_fp32,
                        "rotation_seed": seed,
                        "layer_mse": [ls.mse for ls in stats],
                        "layer_ks_D": [ls.ks_D for ls in stats],
                        "elapsed_s": time.time() - t0,
                    })
                    seed_top1s.append(top1)
                    del model_q
                if len(seed_top1s) > 1:
                    mean, std = np.mean(seed_top1s), np.std(seed_top1s)
                    print(f"  -> {model_name} bits={bits} cb={cb}: "
                          f"mean={mean*100:.2f}%  std={std*100:.3f}pp  "
                          f"(over seeds {args.seeds})")

    print(f"\nDone. Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
