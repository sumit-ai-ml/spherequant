"""LLM benchmark path: causal LM + WikiText-2-style perplexity.

Two entry points:

  - :func:`benchmark_causal_lm` — public Python API. Takes any causal-LM
    ``nn.Module`` plus a tokenizer and a text corpus. Use this when your
    model is a state-dict you've already loaded, or anything that isn't a
    ``AutoModelForCausalLM`` from the Hub.
  - :func:`run` — CLI helper called by ``python -m apexquant.bench``.
    Loads an HF model + HF text dataset, then dispatches to
    ``benchmark_causal_lm``.

The perplexity protocol matches GPTQ/AWQ/QuaRot: concatenate the chosen
text into one corpus, tokenize once, split into non-overlapping chunks of
``seq_len``, take ``exp(mean cross-entropy)``.

Default HF dataset for ``run``: ``wikitext`` config ``wikitext-2-raw-v1``
test split.
"""

from __future__ import annotations

import copy
import gc
import math
import time
from typing import Optional

import torch
import torch.nn as nn

from apexquant.audit import audit
from apexquant.bench._eval import BenchResult
from apexquant.exceptions import ApexQuantPreflightWarning
from apexquant.ptq import quantize_model

DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "test"


def _resolve_split(hf, requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    if hasattr(hf, "keys"):
        for cand in ("test", "validation", "val", "train"):
            if cand in hf.keys():
                return cand
        return list(hf.keys())[0]
    return DEFAULT_SPLIT


@torch.no_grad()
def _perplexity(model, tokenizer, text: str, device: str,
                seq_len: int, max_chunks: Optional[int]) -> tuple[float, int]:
    """Returns (perplexity, n_chunks_used)."""
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    total_tokens = input_ids.shape[1]

    n_chunks = total_tokens // seq_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    if n_chunks == 0:
        chunk = input_ids
        out = model(chunk, labels=chunk)
        return float(math.exp(out.loss.float().item())), 1

    total_loss = 0.0
    total_count = 0
    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len: (i + 1) * seq_len]
        out = model(chunk, labels=chunk)
        total_loss += out.loss.float().item() * (seq_len - 1)
        total_count += seq_len - 1
    return float(math.exp(total_loss / total_count)), n_chunks


def benchmark_causal_lm(
    model: nn.Module,
    tokenizer,
    text: str,
    *,
    model_name: str = "<local>",
    dataset_name: str = "<local>",
    bits_list: list[int] = (4, 8),
    methods: list[str] = ("apexquant", "quarot", "rtn_absmax"),
    codebook: str = "beta",
    rotation_seed: int = 0,
    seq_len: int = 2048,
    max_chunks: Optional[int] = None,
    device: Optional[str] = None,
    preflight: bool = True,
    verbose: bool = True,
) -> list[BenchResult]:
    """Audit, quantize, and benchmark any causal LM on a text corpus.

    The model can be any ``nn.Module`` whose forward call accepts ``(input_ids,
    labels=...)`` and returns an output object with ``.loss`` (the standard HF
    causal-LM signature). The tokenizer must accept ``text`` and return a
    dict-like with ``input_ids``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not text.strip():
        raise SystemExit("Text corpus is empty.")

    if verbose:
        print("\nRunning fan-in audit...")
    audit(model, verbose=verbose)

    results: list[BenchResult] = []

    if verbose:
        print("\nEvaluating reference model...")
    t0 = time.time()
    model_dev = model.to(device)
    ppl, n_chunks = _perplexity(model_dev, tokenizer, text, device,
                                seq_len, max_chunks)
    elapsed = time.time() - t0
    # Determine reference precision from the first parameter's dtype.
    ref_dtype = next(model_dev.parameters()).dtype
    ref_bits = 16 if ref_dtype in (torch.float16, torch.bfloat16) else 32
    ref_label = "fp16" if ref_dtype == torch.float16 else (
        "bf16" if ref_dtype == torch.bfloat16 else "fp32"
    )
    if verbose:
        print(f"  reference ({ref_label}): ppl={ppl:.3f}  chunks={n_chunks}  "
              f"[{elapsed:.0f}s]")
    results.append(BenchResult(
        model=model_name, task="causal_lm", dataset=dataset_name,
        variant="reference", bits=ref_bits, codebook=ref_label,
        n_samples=n_chunks, perplexity=ppl, elapsed_s=elapsed,
    ))
    model = model_dev.cpu()

    for bits in bits_list:
        for method in methods:
            cb = codebook if method in ("apexquant", "baseline") else "symabs_uniform"
            t0 = time.time()
            model_q = copy.deepcopy(model)
            try:
                model_q, _stats = quantize_model(
                    model_q, bits=bits, method=method,
                    codebook=cb, rotation_seed=rotation_seed,
                    preflight=preflight,
                )
            except ApexQuantPreflightWarning as e:
                if verbose:
                    print(f"  {method:11s} bits={bits}  REFUSED by preflight: {e}")
                del model_q
                gc.collect()
                continue
            model_q = model_q.to(device)
            ppl, n_chunks = _perplexity(model_q, tokenizer, text, device,
                                        seq_len, max_chunks)
            elapsed = time.time() - t0
            if verbose:
                print(f"  {method:11s} bits={bits}  cb={cb:16s}  "
                      f"ppl={ppl:10.3f}  [{elapsed:.0f}s]")
            results.append(BenchResult(
                model=model_name, task="causal_lm", dataset=dataset_name,
                variant=method, bits=bits, codebook=cb,
                n_samples=n_chunks, perplexity=ppl, elapsed_s=elapsed,
            ))
            del model_q
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def _load_hf_dataset(dataset_id: str, dataset_config: Optional[str],
                     data_dir: Optional[str]):
    from datasets import load_dataset
    kwargs = {}
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return load_dataset(dataset_id, **kwargs)


def run(
    *,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    dataset_id: Optional[str],
    dataset_config: Optional[str],
    dataset_split: Optional[str],
    data_dir: Optional[str] = None,
    text_col: str,
    bits_list: list[int],
    methods: list[str],
    codebook: str,
    rotation_seed: int,
    seq_len: int,
    max_chunks: Optional[int],
    subset_size: Optional[int],
    device: str,
    preflight: bool,
) -> list[BenchResult]:
    """CLI helper. Loads model + HF dataset and dispatches to
    :func:`benchmark_causal_lm`."""
    if (model_id is None) == (checkpoint_path is None):
        raise SystemExit("Pass exactly one of model_id or checkpoint_path.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_id = dataset_id or DEFAULT_DATASET
    if dataset_id == DEFAULT_DATASET and dataset_config is None:
        dataset_config = DEFAULT_DATASET_CONFIG

    dtype = torch.float16 if device == "cuda" else torch.float32
    if model_id is not None:
        print(f"Loading HF model {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model_name = model_id
    else:
        if tokenizer_id is None:
            raise SystemExit(
                "--checkpoint with --task causal_lm also needs --tokenizer "
                "(an HF tokenizer repo ID or local path matching the model)."
            )
        print(f"Loading checkpoint {checkpoint_path}...")
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise SystemExit(
                "--checkpoint must contain a full nn.Module (torch.save(model, ...)), "
                "not a state_dict. For state-dicts, reconstruct the model in code "
                "and call apexquant.bench.benchmark_causal_lm directly."
            )
        model = obj
        print(f"Loading tokenizer {tokenizer_id}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model_name = checkpoint_path
    model.eval()

    print(f"Loading dataset {dataset_id} (config={dataset_config}, "
          f"data_dir={data_dir})...")
    hf = _load_hf_dataset(dataset_id, dataset_config, data_dir)
    split = _resolve_split(hf, dataset_split)
    hf_split = hf[split] if hasattr(hf, "keys") else hf
    print(f"  using split: {split}  ({len(hf_split)} rows)")

    if text_col not in hf_split.column_names:
        raise SystemExit(
            f"Text column {text_col!r} not in dataset columns "
            f"{hf_split.column_names}. Pass --text-col explicitly."
        )

    if subset_size is not None:
        hf_split = hf_split.select(range(min(subset_size, len(hf_split))))
    text = "\n\n".join(s for s in hf_split[text_col] if s)

    return benchmark_causal_lm(
        model, tokenizer, text,
        model_name=model_name, dataset_name=dataset_id,
        bits_list=bits_list, methods=methods, codebook=codebook,
        rotation_seed=rotation_seed, seq_len=seq_len, max_chunks=max_chunks,
        device=device, preflight=preflight,
    )
