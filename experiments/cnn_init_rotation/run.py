"""Orchestrator: train baseline and H3 models for each seed, then run the PTQ
sweep (bits x {uniform, beta} codebook x {baseline, ApexQuant, H3}).

Writes results/results.jsonl (one row per evaluation with accuracy + AUROC)
and caches raw softmax scores under results/scores/ so new metrics can be
recomputed later without re-running the sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train import TrainConfig, get_dataloaders, train
from apexquant.ptq import (
    quantize_model_baseline,
    quantize_model_apexquant,
    quantize_model_h3,
    eval_with_scores,
    compute_auroc_macro,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_DIR = RESULTS_DIR / "scores"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
LABELS_FILE = RESULTS_DIR / "test_labels.npy"


def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def layer_stats_to_dict(stats):
    return [{"name": s.name, "d": s.d, "mse": s.mse, "ks_D": s.ks_D} for s in stats]


def score_path(variant: str, seed: int, bits: int | None, codebook: str | None) -> Path:
    if bits is None:
        return SCORES_DIR / f"fp32_{variant}_seed{seed}.npy"
    return SCORES_DIR / f"{variant}_seed{seed}_bits{bits}_{codebook}.npy"


def run_ptq_sweep(variant_label: str, model, rotation_seed: int,
                  test_loader, device, bits_list, codebooks, fp32_acc: float,
                  fp32_auroc: float, seed: int, epoch_history):
    """Run full PTQ sweep against a single trained model.

    variant_label drives the quantization path:
      - "baseline": quantize_model_baseline (no rotation)
      - "apexquant":       quantize_model_apexquant (post-hoc rotation on baseline weights)
      - "h3":       quantize_model_h3 (quantize U in a RotatedCNN3)
    """
    for bits in bits_list:
        for codebook in codebooks:
            t0 = time.time()
            if variant_label == "baseline":
                model_q, stats = quantize_model_baseline(model, bits, codebook)
            elif variant_label == "apexquant":
                model_q, stats = quantize_model_apexquant(
                    model, bits, codebook, rotation_seed=rotation_seed
                )
            elif variant_label == "h3":
                model_q, stats = quantize_model_h3(model, bits, codebook)
            else:
                raise ValueError(variant_label)
            acc, scores, labels = eval_with_scores(
                model_q.to(device), test_loader, device
            )
            auroc = compute_auroc_macro(scores, labels)
            # Cache scores so future metrics are cheap
            np.save(score_path(variant_label, seed, bits, codebook), scores)
            row = {
                "variant": variant_label,
                "seed": seed,
                "bits": bits,
                "codebook": codebook,
                "fp32_acc": fp32_acc,
                "fp32_auroc": fp32_auroc,
                "quant_acc": acc,
                "quant_auroc": auroc,
                "acc_drop": fp32_acc - acc,
                "auroc_drop": fp32_auroc - auroc,
                "layer_stats": layer_stats_to_dict(stats),
                "elapsed_s": time.time() - t0,
                "epoch_history": epoch_history,
            }
            append_result(row)
            print(f"    {variant_label:8s} bits={bits}  codebook={codebook:7s}  "
                  f"acc={acc*100:.2f}%  auroc={auroc:.4f}  "
                  f"(Δacc={(fp32_acc-acc)*100:+.2f}pp  "
                  f"Δauroc={(fp32_auroc-auroc):+.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--codebooks", type=str, nargs="+",
                        default=["uniform", "beta"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clear", action="store_true",
                        help="wipe results/ before running")
    args = parser.parse_args()

    if args.clear:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
        for p in SCORES_DIR.glob("*.npy"):
            p.unlink()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}  Bits: {args.bits}  Codebooks: {args.codebooks}  "
          f"Epochs: {args.epochs}")

    _, test_loader = get_dataloaders(cfg.batch_size, cfg.num_workers)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")

        # Train baseline + capture FP32 scores
        print(f"  [train] baseline (CNN3, Kaiming init)")
        res_base = train("baseline", seed=seed, cfg=cfg, device=device)
        fp32_acc_base, scores_base, labels = eval_with_scores(
            res_base["model"].to(device), test_loader, device
        )
        fp32_auroc_base = compute_auroc_macro(scores_base, labels)
        np.save(score_path("fp32_baseline", seed, None, None), scores_base)
        if not LABELS_FILE.exists():
            np.save(LABELS_FILE, labels)
        print(f"    FP32 baseline  acc={fp32_acc_base*100:.2f}%  "
              f"auroc={fp32_auroc_base:.4f}")

        # Train H3 + capture FP32 scores
        print(f"  [train] h3_rotated_basis (RotatedCNN3)")
        res_h3 = train("h3_rotated_basis", seed=seed, cfg=cfg, device=device)
        fp32_acc_h3, scores_h3, _ = eval_with_scores(
            res_h3["model"].to(device), test_loader, device
        )
        fp32_auroc_h3 = compute_auroc_macro(scores_h3, labels)
        np.save(score_path("fp32_h3", seed, None, None), scores_h3)
        print(f"    FP32 H3        acc={fp32_acc_h3*100:.2f}%  "
              f"auroc={fp32_auroc_h3:.4f}")

        # PTQ sweep: baseline quantized directly (control)
        run_ptq_sweep("baseline", res_base["model"], rotation_seed=seed,
                      test_loader=test_loader, device=device,
                      bits_list=args.bits, codebooks=args.codebooks,
                      fp32_acc=fp32_acc_base, fp32_auroc=fp32_auroc_base,
                      seed=seed, epoch_history=res_base["history"])

        # PTQ sweep: baseline with POST-HOC rotation (ApexQuant)
        run_ptq_sweep("apexquant", res_base["model"], rotation_seed=seed,
                      test_loader=test_loader, device=device,
                      bits_list=args.bits, codebooks=args.codebooks,
                      fp32_acc=fp32_acc_base, fp32_auroc=fp32_auroc_base,
                      seed=seed, epoch_history=res_base["history"])

        # PTQ sweep: H3-trained model (quantize U)
        run_ptq_sweep("h3", res_h3["model"], rotation_seed=seed,
                      test_loader=test_loader, device=device,
                      bits_list=args.bits, codebooks=args.codebooks,
                      fp32_acc=fp32_acc_h3, fp32_auroc=fp32_auroc_h3,
                      seed=seed, epoch_history=res_h3["history"])

    print(f"\nDone. Results: {RESULTS_FILE}")
    print(f"Scores cached to:   {SCORES_DIR}")


if __name__ == "__main__":
    main()
