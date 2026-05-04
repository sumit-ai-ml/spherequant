"""CLI entry point for ``python -m apexquant.bench``.

Dispatches between the vision and LLM benchmark paths. Supports two model
sources:

  - ``--hf-model <repo_or_local_path>`` — any HF
    ``AutoModelForImageClassification`` / ``AutoModelForCausalLM`` (Hub
    repo ID *or* a local directory written by ``model.save_pretrained``).
    Task is auto-detected from the model's ``AutoConfig``.
  - ``--checkpoint <path.pt>`` — a full ``nn.Module`` saved with
    ``torch.save(model, path)``. Task isn't introspectable, so pass
    ``--task image_classification`` or ``--task causal_lm``. For LLMs you
    also need ``--tokenizer`` (a tokenizer repo ID or local path).

For state-dict checkpoints, use the Python API directly — see
``apexquant.bench.benchmark_image_classifier`` and
``benchmark_causal_lm``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _detect_task(model_id: str) -> str:
    """Return 'image_classification' or 'causal_lm' from the HF AutoConfig."""
    try:
        from transformers import AutoConfig
    except ImportError:
        raise SystemExit(
            "apexquant.bench requires the transformers and datasets packages.\n"
            "Install with: pip install transformers datasets"
        )

    cfg = AutoConfig.from_pretrained(model_id)
    archs = getattr(cfg, "architectures", None) or []
    archs_lc = " ".join(a.lower() for a in archs)

    if "forimageclassification" in archs_lc:
        return "image_classification"
    if "forcausallm" in archs_lc or "lmhead" in archs_lc:
        return "causal_lm"

    if hasattr(cfg, "image_size") and not hasattr(cfg, "vocab_size"):
        return "image_classification"
    if hasattr(cfg, "vocab_size"):
        return "causal_lm"

    raise SystemExit(
        f"Could not auto-detect task for {model_id!r}. "
        f"architectures={archs}. apexquant.bench supports image classification "
        f"and causal LM only."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m apexquant.bench",
        description="Audit, quantize, and benchmark a model end-to-end.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-model",
                     help="HF model repo ID or local save_pretrained directory.")
    src.add_argument("--checkpoint",
                     help="Path to a full nn.Module saved with torch.save(model, ...). "
                          "For state-dicts, use the Python API "
                          "(apexquant.bench.benchmark_image_classifier / _causal_lm).")
    p.add_argument("--task", choices=["image_classification", "causal_lm"],
                   default=None,
                   help="Required with --checkpoint; auto-detected with --hf-model.")
    p.add_argument("--tokenizer",
                   help="Tokenizer repo ID or local path. Required with "
                        "--checkpoint --task causal_lm.")
    p.add_argument("--image-size", type=int, default=224,
                   help="Vision --checkpoint only: input resolution for the "
                        "torchvision-style transform (Resize/CenterCrop/Normalize).")

    p.add_argument("--hf-dataset", default=None,
                   help="HF dataset repo ID or builder name (e.g. 'imagefolder', "
                        "'csv', 'parquet'). Defaults: imagenet-1k val for vision, "
                        "wikitext-2 for LLM.")
    p.add_argument("--dataset-config", default=None,
                   help="Optional dataset config name (e.g. 'wikitext-2-raw-v1').")
    p.add_argument("--dataset-split", default=None,
                   help="Dataset split. Defaults to validation/test if available.")
    p.add_argument("--data-dir", default=None,
                   help="Local data directory passed to load_dataset(data_dir=...). "
                        "Use with --hf-dataset imagefolder for a local folder of "
                        "images-in-class-subdirs.")

    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--methods", nargs="+",
                   default=["apexquant", "quarot", "rtn_absmax"],
                   choices=["apexquant", "quarot", "rtn_absmax", "baseline"])
    p.add_argument("--codebook", default="beta", choices=["beta", "uniform"],
                   help="Codebook for apexquant/baseline. Ignored by "
                        "quarot/rtn_absmax.")
    p.add_argument("--rotation-seed", type=int, default=0)
    p.add_argument("--subset-size", type=int, default=None,
                   help="Limit eval to first N samples (smoke tests).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048,
                   help="LLM only: chunk length for perplexity.")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="LLM only: cap perplexity chunks.")
    p.add_argument("--image-col", default=None,
                   help="Vision: dataset column with the image (auto-detected if omitted).")
    p.add_argument("--label-col", default=None,
                   help="Vision: dataset column with the integer label (auto-detected).")
    p.add_argument("--text-col", default="text",
                   help="LLM: dataset column with the text body.")
    p.add_argument("--device", default=None, help="cuda / cpu. Default: auto.")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip BAD-verdict refusal in quantize_model.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path for JSONL results.")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.hf_model is not None:
        task = args.task or _detect_task(args.hf_model)
    else:
        if args.task is None:
            raise SystemExit("--checkpoint requires --task image_classification or --task causal_lm.")
        task = args.task
    print(f"Task: {task}  Device: {device}")

    if task == "image_classification":
        from apexquant.bench import vision
        results = vision.run(
            model_id=args.hf_model,
            checkpoint_path=args.checkpoint,
            image_size=args.image_size,
            dataset_id=args.hf_dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            data_dir=args.data_dir,
            image_col=args.image_col,
            label_col=args.label_col,
            bits_list=args.bits,
            methods=args.methods,
            codebook=args.codebook,
            rotation_seed=args.rotation_seed,
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            preflight=not args.no_preflight,
        )
    else:
        from apexquant.bench import llm
        results = llm.run(
            model_id=args.hf_model,
            checkpoint_path=args.checkpoint,
            tokenizer_id=args.tokenizer,
            dataset_id=args.hf_dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            data_dir=args.data_dir,
            text_col=args.text_col,
            bits_list=args.bits,
            methods=args.methods,
            codebook=args.codebook,
            rotation_seed=args.rotation_seed,
            seq_len=args.seq_len,
            max_chunks=args.max_chunks,
            subset_size=args.subset_size,
            device=device,
            preflight=not args.no_preflight,
        )

    from apexquant.bench._eval import format_summary, write_jsonl
    print(format_summary(results))
    if args.out is not None:
        write_jsonl(results, args.out)
        print(f"Wrote {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
