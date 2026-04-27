"""Orchestrator for LLM PTQ sweep.

For each model × bit width × method, quantize the weights in place
(deep-copying the model first) and evaluate WikiText-2 perplexity.

Methods:
  - "rtn"    : baseline round-to-nearest per-channel absmax (no rotation)
  - "quarot" : rotation + RTN absmax (our QuaRot-style baseline)
  - "h2_beta": rotation + per-row L2 normalize + Beta codebook (ours)

Perplexity is the standard WikiText-2 metric used by GPTQ, AWQ, QuaRot.
Lower is better. FP16 reference is the target.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent
_CNN_EXP = _THIS.parent / "cnn_init_rotation"
sys.path.insert(0, str(_CNN_EXP))

# Reuse the core primitives
from ptq import (  # noqa: E402
    quantize_model_baseline,   # per-row L2 + codebook (CNN-style)
    quantize_model_rtn_absmax, # per-channel absmax uniform, no rotation (LLM RTN)
    quantize_model_h2,
    quantize_model_quarot,
)

from models import load_model, model_size_at_bits
from data import perplexity, get_wikitext2_text


RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.jsonl"


def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_one(model, tokenizer, model_id: str, variant: str, bits: int,
            codebook: str, device: str, text: str, rotation_seed: int,
            seq_len: int, max_chunks=None, size_info=None):
    t0 = time.time()
    # Deep-copy so the original model stays available for the next variant
    # (saves re-downloading, re-loading).
    model_q = copy.deepcopy(model)

    if variant == "rtn_absmax":
        # Canonical LLM RTN: per-channel absmax, no rotation. The real baseline
        # in the LLM quantization literature (GPTQ, AWQ, QuaRot all compare
        # against this).
        model_q, stats = quantize_model_rtn_absmax(model_q, bits)
    elif variant == "rtn":
        # Per-row L2 + codebook (CNN paper baseline). Not a real LLM baseline.
        # Kept as a diagnostic showing why L2-normalize + uniform breaks for LLMs.
        model_q, stats = quantize_model_baseline(model_q, bits, codebook)
    elif variant == "quarot":
        model_q, stats = quantize_model_quarot(model_q, bits,
                                               rotation_seed=rotation_seed)
    elif variant == "h2":
        model_q, stats = quantize_model_h2(model_q, bits, codebook,
                                           rotation_seed=rotation_seed)
    else:
        raise ValueError(f"unknown variant {variant}")

    ppl = perplexity(model_q, tokenizer, device=device, seq_len=seq_len,
                     max_chunks=max_chunks, text=text)
    elapsed = time.time() - t0

    row = {
        "model": model_id,
        "variant": variant,
        "bits": bits,
        "codebook": codebook,
        "perplexity": ppl,
        "size_mb": size_info["quantized_mb"] if size_info else None,
        "compression_ratio": size_info["ratio_vs_fp16"] if size_info else None,
        "elapsed_s": elapsed,
        "rotation_seed": rotation_seed,
        "layer_mse_mean": float(sum(s.mse for s in stats) / max(1, len(stats))),
        "n_layers_quantized": len(stats),
    }
    append_result(row)
    print(f"  {variant:8s} bits={bits}  cb={codebook:8s}  "
          f"ppl={ppl:8.3f}  size={size_info['quantized_mb']:.1f}MB "
          f"({size_info['ratio_vs_fp16']:.2f}x vs fp16)  [{elapsed:.0f}s]")

    del model_q
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sweep_model(model_id: str, bits_list, codebooks, device: str,
                rotation_seed: int = 0, seq_len: int = 2048,
                max_chunks=None):
    print(f"\n=== {model_id} ===")
    model, tokenizer = load_model(model_id)
    info = model_size_at_bits(model, None)
    print(f"  params: {info['n_params_total']:,}  "
          f"quantizable: {info['n_weights_quant']:,}  "
          f"fp16 size: {info['fp16_mb']:.1f} MB")

    # Load WikiText-2 text once
    text = get_wikitext2_text("test")
    print("  WikiText-2 test set loaded.")

    # FP16 reference
    t0 = time.time()
    ppl_fp16 = perplexity(model, tokenizer, device=device, seq_len=seq_len,
                          max_chunks=max_chunks, text=text)
    elapsed = time.time() - t0
    print(f"  FP16: ppl={ppl_fp16:.3f}  [{elapsed:.0f}s]")
    append_result({
        "model": model_id,
        "variant": "fp16",
        "bits": 16,
        "codebook": "fp16",
        "perplexity": ppl_fp16,
        "size_mb": info["fp16_mb"],
        "compression_ratio": 1.0,
        "elapsed_s": elapsed,
    })

    # Move model back to CPU between quantization rounds (deepcopy on GPU
    # doubles memory). The first deepcopy moves everything to CPU with
    # copy.deepcopy on the GPU state; safer to leave the original on CPU.
    model = model.cpu()

    for bits in bits_list:
        s = model_size_at_bits(model, bits)
        # 1. Canonical LLM RTN: per-channel absmax, no rotation
        run_one(model, tokenizer, model_id, "rtn_absmax", bits, "symabs_uniform",
                device, text, rotation_seed, seq_len, max_chunks, s)
        # 2. QuaRot-RTN (rotation + per-channel absmax uniform)
        run_one(model, tokenizer, model_id, "quarot", bits, "symabs_uniform",
                device, text, rotation_seed, seq_len, max_chunks, s)
        # 3. H2 Beta (ours)
        run_one(model, tokenizer, model_id, "h2", bits, "beta",
                device, text, rotation_seed, seq_len, max_chunks, s)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--codebooks", nargs="+", default=["uniform", "beta"],
                   help="(codebooks only used for baseline RTN; ignored here)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="limit chunks per eval for smoke tests")
    p.add_argument("--rotation-seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--clear", action="store_true")
    args = p.parse_args()

    if args.clear and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}  Bits: {args.bits}")

    for model_id in args.models:
        sweep_model(model_id, args.bits, args.codebooks, device,
                    rotation_seed=args.rotation_seed,
                    seq_len=args.seq_len,
                    max_chunks=args.max_chunks)

    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
