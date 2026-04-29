"""SphereQuant post-training quantization for Llama 3 8B and 70B on Hendrix.

Key differences from the laptop-scale run.py in the parent folder:

1. No deepcopy. Large models cannot afford two copies in GPU memory. Instead
   we snapshot the original weights to CPU once at load time, quantize
   in-place into the model, evaluate, and restore from the CPU snapshot
   between methods.

2. device_map="auto" for the 70B model, which shards layers across all
   visible GPUs.

3. lm-eval-harness integration. In addition to WikiText-2 perplexity, each
   quantized configuration is evaluated on six standard zero-shot tasks.

4. Layer-by-layer quantization with explicit GPU staging. For each
   nn.Linear, we move its weight to CPU (a few MB per layer), apply the
   numpy quantization functions, and write the result back to the layer's
   native device. This avoids host-side tensors of tens of GB.

Methods implemented: RTN-absmax, QuaRot-RTN, SphereQuant+Beta. All training-free.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the core PTQ primitives from the CNN experiment folder.
_THIS = Path(__file__).resolve().parent

from spherequant.ptq import (  # noqa: E402
    per_row_quantize,
    make_rotation,
    apply_rotation,
    SRHTRotation,
)

RESULTS_DIR = _THIS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "llama3_results.jsonl"


# -------------------------------------------------------------------------
# Weight snapshot and restore without deepcopy
# -------------------------------------------------------------------------

def snapshot_linear_weights(model: nn.Module) -> dict:
    """Save a CPU copy of every nn.Linear weight for later restore.

    Keeps dtype but always lives on CPU. For Llama 3 70B this is ~140 GB.
    """
    print("  snapshotting nn.Linear weights to CPU...")
    t0 = time.time()
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            snap[name] = m.weight.detach().cpu().clone()
    print(f"    {len(snap)} layers, took {time.time()-t0:.1f}s, "
          f"snapshot size = {sum(v.numel()*v.element_size() for v in snap.values())/(1024**3):.2f} GB")
    return snap


def restore_from_snapshot(model: nn.Module, snap: dict):
    """Write snapshot back into model weights in place."""
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and name in snap:
                m.weight.copy_(snap[name].to(m.weight.device, m.weight.dtype))


# -------------------------------------------------------------------------
# In-place quantization (no deepcopy)
# -------------------------------------------------------------------------

def _layer_to_numpy(layer: nn.Linear) -> np.ndarray:
    return layer.weight.detach().cpu().numpy().astype(np.float32)


def _write_numpy_back(layer: nn.Linear, W_np: np.ndarray):
    with torch.no_grad():
        layer.weight.copy_(
            torch.from_numpy(W_np).to(layer.weight.dtype).to(layer.weight.device)
        )


def quantize_model_inplace(model: nn.Module, method: str, bits: int,
                           rotation_seed: int = 0):
    """Quantize every nn.Linear in model in place. Method in
    {rtn_absmax, quarot, sq}."""
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    t0 = time.time()
    for idx, (name, layer) in enumerate(layers):
        W = _layer_to_numpy(layer)
        N, d = W.shape

        if method == "rtn_absmax":
            W_rec = _rtn_absmax(W, bits)
        elif method == "quarot":
            W_rec = _quarot(W, bits, rotation_seed, idx)
        elif method == "spherequant":
            W_rec = _h2(W, bits, rotation_seed, idx)
        else:
            raise ValueError(method)

        _write_numpy_back(layer, W_rec)
        if (idx + 1) % 32 == 0:
            print(f"    quantized {idx+1}/{len(layers)} layers "
                  f"({time.time()-t0:.1f}s)")
    print(f"  quantization done: {len(layers)} layers in {time.time()-t0:.1f}s")


def _rtn_absmax(W: np.ndarray, bits: int) -> np.ndarray:
    scale = np.max(np.abs(W), axis=1, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-12, 1.0, scale)
    levels = (1 << (bits - 1)) - 1 if bits > 1 else 1
    W_int = np.round(W / scale * levels).clip(-levels, levels)
    return (W_int * scale / levels).astype(np.float32)


def _quarot(W: np.ndarray, bits: int, seed: int, layer_idx: int) -> np.ndarray:
    _, d = W.shape
    rot = make_rotation(d, seed * 1000 + layer_idx + 1, "srht")
    U = apply_rotation(W, rot)
    scale = np.max(np.abs(U), axis=1, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-12, 1.0, scale)
    levels = (1 << (bits - 1)) - 1 if bits > 1 else 1
    U_int = np.round(U / scale * levels).clip(-levels, levels)
    U_rec = (U_int * scale / levels).astype(np.float32)
    if isinstance(rot, SRHTRotation):
        return rot.inverse(U_rec).astype(np.float32)
    return (U_rec @ rot).astype(np.float32)


def _h2(W: np.ndarray, bits: int, seed: int, layer_idx: int) -> np.ndarray:
    _, d = W.shape
    rot = make_rotation(d, seed * 1000 + layer_idx + 1, "srht")
    U = apply_rotation(W, rot)
    U_rec, _ = per_row_quantize(U, bits, "beta")
    if isinstance(rot, SRHTRotation):
        return rot.inverse(U_rec).astype(np.float32)
    return (U_rec @ rot).astype(np.float32)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def wikitext2_perplexity(model, tokenizer, seq_len: int = 2048) -> float:
    """Full WikiText-2 raw test perplexity."""
    from datasets import load_dataset
    import math
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n = input_ids.shape[1]
    n_chunks = n // seq_len
    total_loss, total_count = 0.0, 0
    model.eval()
    for i in range(n_chunks):
        chunk = input_ids[:, i*seq_len:(i+1)*seq_len]
        # Route to the same device as the embedding (for device_map="auto")
        chunk = chunk.to(model.device if hasattr(model, "device") else "cuda")
        out = model(chunk, labels=chunk)
        total_loss += out.loss.float().item() * (seq_len - 1)
        total_count += seq_len - 1
    return float(math.exp(total_loss / total_count))


def lm_eval_tasks(model, tokenizer, tasks=None, batch_size: int = 1) -> dict:
    """Wrap model for lm-eval-harness and run a fixed set of zero-shot tasks."""
    if tasks is None:
        tasks = ["lambada_openai", "hellaswag", "piqa",
                 "winogrande", "arc_easy", "arc_challenge"]
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        print(f"  lm-eval not installed, skipping zero-shot tasks ({e})")
        return {}

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, batch_size=batch_size)
    results = {}
    for task in tasks:
        if task in out["results"]:
            r = out["results"][task]
            # Report the primary accuracy metric if present, else first metric
            for key in ("acc,none", "acc_norm,none", "perplexity,none"):
                if key in r:
                    results[task] = r[key]
                    break
    return results


# -------------------------------------------------------------------------
# Sweep orchestrator
# -------------------------------------------------------------------------

def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def model_size_mb_fp16(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 2 / (1024 ** 2)


def quantized_size_mb(model: nn.Module, bits: int) -> float:
    n_w, n_rows, n_other = 0, 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n_w += m.weight.numel()
            n_rows += m.weight.shape[0]
    total = sum(p.numel() for p in model.parameters())
    n_other = total - n_w
    code = n_w * bits / 8
    norms = n_rows * 4
    other = n_other * 2  # LN and small params stay fp16
    return (code + norms + other) / (1024 ** 2)


def sweep(model_id: str, bits_list, methods, run_lm_eval: bool,
          rotation_seed: int = 0):
    print(f"\n=== {model_id} ===")

    # Loading: device_map="auto" shards across all visible GPUs
    print(f"  loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"    loaded in {time.time()-t0:.1f}s. "
          f"FP16 size: {model_size_mb_fp16(model):.0f} MB")

    # FP16 reference
    t0 = time.time()
    ppl_fp16 = wikitext2_perplexity(model, tokenizer)
    print(f"  FP16: ppl={ppl_fp16:.3f}  [{time.time()-t0:.0f}s]")
    row = {"model": model_id, "variant": "fp16", "bits": 16, "codebook": "fp16",
           "perplexity": ppl_fp16, "size_mb": model_size_mb_fp16(model),
           "compression_ratio": 1.0, "elapsed_s": time.time() - t0}
    if run_lm_eval:
        t1 = time.time()
        row["lm_eval"] = lm_eval_tasks(model, tokenizer)
        print(f"  FP16 lm-eval: {row['lm_eval']}  [{time.time()-t1:.0f}s]")
    append_result(row)

    # Snapshot for restore between methods
    snap = snapshot_linear_weights(model)

    for bits in bits_list:
        size_q = quantized_size_mb(model, bits)
        for method in methods:
            t0 = time.time()
            print(f"\n  [{method} @ {bits}-bit]")
            quantize_model_inplace(model, method, bits, rotation_seed=rotation_seed)
            ppl = wikitext2_perplexity(model, tokenizer)
            print(f"    ppl={ppl:.3f}  size={size_q:.0f}MB  [{time.time()-t0:.0f}s]")

            row = {
                "model": model_id,
                "variant": method,
                "bits": bits,
                "codebook": "beta" if method == "spherequant" else "symabs_uniform",
                "perplexity": ppl,
                "size_mb": size_q,
                "compression_ratio": model_size_mb_fp16(model) / size_q,
                "rotation_seed": rotation_seed,
                "elapsed_s": time.time() - t0,
            }
            if run_lm_eval:
                t1 = time.time()
                row["lm_eval"] = lm_eval_tasks(model, tokenizer)
                print(f"    lm-eval: {row['lm_eval']}  [{time.time()-t1:.0f}s]")
            append_result(row)

            # Restore for next method
            print("  restoring from snapshot...")
            restore_from_snapshot(model, snap)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del snap, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model id, e.g. meta-llama/Meta-Llama-3-8B")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--methods", nargs="+",
                   default=["rtn_absmax", "quarot", "spherequant"])
    p.add_argument("--lm-eval", action="store_true",
                   help="also run lm-eval-harness zero-shot tasks")
    p.add_argument("--rotation-seed", type=int, default=0)
    args = p.parse_args()

    sweep(args.model, args.bits, args.methods, run_lm_eval=args.lm_eval,
          rotation_seed=args.rotation_seed)
    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
